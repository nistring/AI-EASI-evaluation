import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl
from torch.distributions.distribution import Distribution

from .model_utils import *
from typing import List, Tuple, Optional


class _HierarchicalCore(nn.Module):
    def __init__(
        self,
        latent_dims,
        in_channels,
        channels_per_block,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        name="HierarchicalDecoderDist",
    ):
        super(_HierarchicalCore, self).__init__()
        self._latent_dims = latent_dims
        self._channels_per_block = channels_per_block
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self.in_channels = in_channels
        self._blocks_per_level = blocks_per_level

        if down_channels_per_block is None:
            self._dowm_channels_per_block = channels_per_block
        else:
            self._dowm_channels_per_block = down_channels_per_block
        self._name = name
        self.num_levels = len(self._channels_per_block)
        self.num_latent_levels = len(self._latent_dims)

        # encoder
        encoder = []
        for level in range(self.num_levels):
            for _ in range(self._blocks_per_level):
                encoder.append(
                    Res_block(
                        in_channels=self.in_channels,
                        out_channels=self._channels_per_block[level],
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                    )
                )
                self.in_channels = self._channels_per_block[level]
            if level != self.num_levels - 1:
                encoder.append(Resize_down())
        self.encoder = nn.Sequential(*encoder)

        # decoder
        decoder = []
        channels = self._channels_per_block[-1]
        for level in range(self.num_latent_levels):
            latent_dim = self._latent_dims[level]
            self.in_channels = 3 * latent_dim + self._channels_per_block[-2 - level]
            decoder.append(nn.Conv2d(channels, 2 * latent_dim, kernel_size=(1, 1), padding=0))
            decoder.append(Resize_up())
            for _ in range(self._blocks_per_level):
                decoder.append(
                    Res_block(
                        in_channels=self.in_channels,
                        out_channels=self._channels_per_block[-2 - level],
                        n_down_channels=self._channels_per_block[-2 - level],
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                    )
                )

                self.in_channels = self._channels_per_block[-2 - level]
            channels = self._channels_per_block[-2 - level]
        self.decoder = nn.Sequential(*decoder)
        self.decoder.apply(init_weights_orthogonal_normal)

    def forward(
        self,
        inputs: torch.Tensor,
        z_q: Optional[List[torch.Tensor]] = None,
        skip_encoder: Optional[List[torch.Tensor]] = None,
        mean: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        if skip_encoder is not None:
            encoder_outputs = skip_encoder
        else:
            encoder_features = inputs
            encoder_outputs = []

            for i, encoder in enumerate(self.encoder, start=1):
                encoder_features = encoder(encoder_features)
                if i % (self._blocks_per_level + 1) == self._blocks_per_level:
                    encoder_outputs.append(encoder_features)
                
        decoder_features = encoder_outputs[-1]
        decoder_output_lo = decoder_features

        distributions, used_latents = [], []
        if torch.jit.isinstance(mean, bool):
            mean = [mean] * self.num_latent_levels

        for i, decoder in enumerate(self.decoder):
            q, r = divmod(i, self._blocks_per_level + 2)
            if r == 0:
                decoder_features = decoder(decoder_features)

                mu = decoder_features[:, : self._latent_dims[q]]
                log_sigma = F.softplus(decoder_features[:, self._latent_dims[q] :]) + 1.192e-07

                # For training
                dist = Normal(loc=mu, scale=log_sigma)
                z = dist.sample()
                if mean[q]:
                    z = dist.mean
                if z_q:
                    z = z_q[q]
                distributions.append(dist)

                # For compiling
                # z = torch.normal(mu, log_sigma)
                
                used_latents.append(z)

                decoder_output_lo = torch.concat([z, decoder_features], dim=1)
            elif r == 1:
                decoder_output_hi = decoder(decoder_output_lo)
                decoder_features = torch.concat([decoder_output_hi, encoder_outputs[::-1][q + 1]], dim=1)
            else:
                decoder_features = decoder(decoder_features)

        return (encoder_outputs, decoder_features, distributions, used_latents)


class _StitchingDecoder(nn.Module):
    def __init__(
        self,
        latent_dims,
        in_channels,
        channels_per_block,
        num_classes,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        name="StitchingDecoder",
    ):
        super(_StitchingDecoder, self).__init__()
        self._latent_dims = latent_dims
        self.in_channels = in_channels
        self._channels_per_block = channels_per_block
        self._num_classes = num_classes
        self._activation_fn = activation_fn
        self._convs_per_block = convs_per_block
        self._blocks_per_level = blocks_per_level
        if down_channels_per_block is None:
            down_channels_per_block = channels_per_block
        self._down_channels_per_block = down_channels_per_block
        self.num_latent = len(self._latent_dims)
        self.start_level = self.num_latent + 1
        self.num_levels = len(self._channels_per_block)

        decoder = []
        for level in range(self.start_level, self.num_levels, 1):
            decoder.append(Resize_up())
            for _ in range(self._blocks_per_level):
                decoder.append(
                    Res_block(
                        in_channels=self.in_channels,
                        out_channels=self._channels_per_block[::-1][level],
                        n_down_channels=int(self._down_channels_per_block[::-1][level]),
                        activation_fn=self._activation_fn,
                        convs_per_block=self._convs_per_block,
                    )
                )
                self.in_channels = self._channels_per_block[::-1][level]
            self.in_channels = self._channels_per_block[::-1][level] + self.in_channels // 2
        self.decoder = nn.Sequential(*decoder)
        self.decoder.apply(init_weights_orthogonal_normal)

        self.last_layer = nn.Conv2d(
            in_channels=self._channels_per_block[::-1][level], out_channels=self._num_classes, kernel_size=(1, 1), padding=0
        )
        self.last_layer.apply(init_weights_orthogonal_normal)

    def forward(self, encoder_features: List[torch.Tensor], decoder_features: torch.Tensor) -> torch.Tensor:
        start_level = self.start_level
        for i, decoder in enumerate(self.decoder):
            decoder_features = decoder(decoder_features)
            if i % (self._blocks_per_level + 1) == 0:
                encoder_feature = encoder_features[::-1][start_level]
                decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
                start_level += 1

        return self.last_layer(decoder_features)


class HierarchicalProbUNet(nn.Module):
    def __init__(
        self,
        latent_dims=(1, 1, 1, 1),
        in_channels=3,
        channels_per_block=None,
        num_classes=6,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        loss_kwargs=None,
        num_cuts=3,
    ):
        super(HierarchicalProbUNet, self).__init__()

        base_channels = 24
        default_channels_per_block = (
            base_channels,
            2 * base_channels,
            4 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
            8 * base_channels,
        )
        if channels_per_block is None:
            channels_per_block = default_channels_per_block
        if down_channels_per_block is None:
            down_channels_per_block = tuple([i / 2 for i in default_channels_per_block])

        self.prior = _HierarchicalCore(
            latent_dims=latent_dims,
            in_channels=in_channels,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="prior",
        )

        self.posterior = _HierarchicalCore(
            latent_dims=latent_dims,
            in_channels=in_channels + num_classes + 1,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="posterior",
        )

        self.f_comb = _StitchingDecoder(
            latent_dims=latent_dims,
            in_channels=channels_per_block[-len(latent_dims) - 1] + channels_per_block[-len(latent_dims) - 2],
            channels_per_block=channels_per_block,
            num_classes=num_classes + 1,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="f_comb",
        )

        self._loss_kwargs = loss_kwargs

        if self._loss_kwargs["type"] == "geco":
            """
            Refer to https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/optimization_constraints.py
            """
            self._lagmul = nn.Parameter(torch.Tensor([self._loss_kwargs["beta"]]))
            self._lagmul.register_hook(lambda grad: -grad * self._loss_kwargs["rate"])
            self.softplus = nn.Softplus()

        self._alpha = nn.Parameter(torch.Tensor([self._loss_kwargs["alpha"]]), requires_grad=False)
        self.cutpoints = nn.Parameter(
            ((torch.arange(1, num_cuts, dtype=torch.float32, requires_grad=True)).expand((num_classes, num_cuts - 1))) / self._alpha
        )
        self._num_classes = num_classes
        self._num_cuts = num_cuts

    def forward(self, seg: torch.Tensor, img: torch.Tensor, mean: bool):
        _q_sample = self.posterior(inputs=torch.concat([seg, img], dim=1), mean=mean, z_q=[])
        _p_sample_z_q = self.prior(inputs=img, z_q=_q_sample[3])  # used_latents
        return _q_sample, _p_sample_z_q

    @torch.jit.export
    def sample(self, img: torch.Tensor, mc_n: int, mean: bool = False):

        with torch.no_grad():
            encoder_features, decoder_features, _, _ = self.prior(inputs=img, mean=mean)
            logits = self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features).unsqueeze(0)

            for i in range(mc_n - 1):
                encoder_features, decoder_features, _, _ = self.prior(inputs=img, mean=mean, skip_encoder=encoder_features)
                logits = torch.cat(
                    (logits, self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features).unsqueeze(0)),
                    dim=0,
                )
            # N x B x H x W x C
            logits = logits.permute(0, 1, 3, 4, 2).float()
            return torch.sigmoid(logits[:, :, :, :, 0:1]), logits[:, :, :, :, 1:]

    def reconstruct(self, seg: torch.Tensor, img: torch.Tensor, mean: bool):
        _q_sample, _p_sample_z_q = self.forward(seg, img, mean)
        encoder_features, decoder_features, _, _ = _p_sample_z_q
        return _q_sample, _p_sample_z_q, self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(
        self, seg: torch.Tensor, img: torch.Tensor, lesion_area: torch.Tensor, skin_area: Optional[torch.Tensor] = None, mean: bool = False
    ):
        _q_sample, _p_sample_z_q, logits = self.reconstruct(torch.cat((lesion_area, seg), dim=1), img, mean=mean)

        device = logits.get_device()
        logits = logits.permute(0, 2, 3, 1)
        seg = seg.permute(0, 2, 3, 1)

        logits = torch.reshape(logits, (-1, self._num_classes + 1))
        y_true = torch.reshape(seg, (-1, self._num_classes))
        lesion_area = lesion_area.reshape(
            -1,
        ).float()

        if skin_area is None:
            skin_area = torch.ones(
                y_true.shape[0],
            ).to(device)
        else:
            skin_area = torch.reshape(skin_area, (-1,)).float()

        # link_mat : N x C x (1, 1, num_cuts - 1)
        # y_true : N x C
        # nll : (N x C x 1)

        batch_size = seg.shape[0]
        link_mat_0, link_mat_1, link_mat = log_cumulative(self.cutpoints * self._alpha, logits[:, 1:])
        y_true = F.one_hot(y_true)
        nll = -torch.log(
            link_mat_0 * y_true[:, :, [0]] + link_mat_1 * y_true[:, :, [1]] + (link_mat * y_true[:, :, 2:]).sum(-1, keepdim=True)
        ).reshape(batch_size, -1, self._num_classes)

        xe = F.binary_cross_entropy_with_logits(logits[:, 0], lesion_area, reduction="none").reshape(batch_size, -1)

        lesion_area = torch.reshape(lesion_area, shape=(batch_size, -1, 1))
        skin_area = torch.reshape(skin_area, shape=(batch_size, -1))

        return (
            _q_sample,
            _p_sample_z_q,
            {
                "mean": (lesion_area * nll).sum() / lesion_area.sum() + (skin_area * xe).sum() / skin_area.sum(),
                "sum": ((lesion_area * nll).sum((1, 2)) + (skin_area * xe).sum(dim=1)).mean(0),
                "mask": skin_area,
            },
        )


    def kl_divergence(self, seg: torch.Tensor, img: torch.Tensor, q_dists: torch.Tensor, p_dists: torch.Tensor):
        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = kl.kl_divergence(p, q)
            kl_per_instance = kl_per_pixel.sum(dim=(1, 2))
            kl_dict[level] = kl_per_instance.mean()

        return kl_dict

    def sum_loss(
        self,
        seg: torch.Tensor,
        img: torch.Tensor,
        lesion_area: torch.Tensor,
        skin_area: Optional[torch.Tensor] = None,
        mean: bool = False,
    ):
        summaries = {}

        # rec loss
        _q_sample, _p_sample_z_q, rec_loss = self.rec_loss(seg, img, lesion_area, skin_area, mean)
        summaries["rec_loss_mean"] = rec_loss["mean"]

        # kl loss
        kl_dict = self.kl_divergence(seg, img, _q_sample[2], _p_sample_z_q[2])
        kl_sum = torch.stack([kl for _, kl in kl_dict.items()], dim=-1).sum()
        summaries["kl_sum"] = kl_sum
        for level, kl in kl_dict.items():
            summaries["kl_{}".format(level)] = kl

        mask_sum_per_instance = rec_loss["mask"].sum(dim=-1)
        num_valid_pixels = mask_sum_per_instance.mean()

        # ELBO
        if self._loss_kwargs["type"] == "elbo":
            loss = rec_loss["sum"] + self._loss_kwargs["beta"] * kl_sum

        # GECO
        elif self._loss_kwargs["type"] == "geco":
            reconstruction_threshold = self._loss_kwargs["kappa"] * num_valid_pixels
            rec_constraint = rec_loss["sum"] - reconstruction_threshold
            lagmul = self.softplus(self._lagmul) ** 2
            loss = lagmul * rec_constraint + kl_sum
            summaries["lagmul"] = self._lagmul
        else:
            raise NotImplementedError("Loss type {} not implemeted!".format(self._loss_kwargs["type"]))

        summaries["loss"] = loss
        return dict(supervised_loss=loss, summaries=summaries)
