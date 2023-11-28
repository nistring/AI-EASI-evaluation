import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from .model_utils import *
from tqdm import tqdm

eps = torch.finfo(torch.float32).eps


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

    def forward(self, inputs, mean=False, z_q=None, skip_encoder=False):
        if skip_encoder:
            encoder_outputs = inputs
        else:
            encoder_features = inputs
            encoder_outputs = []

            count = 0
            for encoder in self.encoder:
                encoder_features = encoder(encoder_features)
                if type(encoder) == Res_block:
                    count += 1
                if count == self._blocks_per_level:
                    encoder_outputs.append(encoder_features)
                    count = 0

        decoder_features = encoder_outputs[-1]

        distributions, used_latents = [], []
        if isinstance(mean, bool):
            mean = [mean] * self.num_latent_levels

        i, j = 0, 0
        features = dict()

        for decoder in self.decoder:
            if type(decoder) == nn.Conv2d:
                decoder_features = decoder(decoder_features)

                mu = decoder_features[:, : self._latent_dims[i]]
                log_sigma = F.softplus(decoder_features[:, self._latent_dims[i] :]) + eps

                norm = Normal(loc=mu, scale=log_sigma)
                dist = Independent(norm, 1)
                distributions.append(dist)

                z = dist.sample()
                if mean[i]:
                    z = dist.mean
                if z_q is not None:
                    if i < len(z_q):
                        z = z_q[i]
                used_latents.append(z)

                decoder_output_lo = torch.concat([z, decoder_features], dim=1)
                i += 1
            elif type(decoder) == Resize_up:
                decoder_output_hi = decoder(decoder_output_lo)
                decoder_features = torch.concat([decoder_output_hi, encoder_outputs[::-1][j + 1]], dim=1)
                j += 1
            else:
                decoder_features = decoder(decoder_features)

        features["decoder_features"] = decoder_features
        features["encoder_features"] = encoder_outputs
        features["distributions"] = distributions
        features["used_latents"] = used_latents

        return features


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

    def forward(self, encoder_features, decoder_features):
        start_level = self.start_level
        for decoder in self.decoder:
            decoder_features = decoder(decoder_features)
            if type(decoder) == Resize_up:
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

    def forward(self, seg, img, mean):
        self._q_sample = self.posterior(torch.concat([seg, img], dim=1), mean=mean)
        self._p_sample_z_q = self.prior(img, z_q=self._q_sample["used_latents"])
        return

    def sample(self, img, mc_n: int, mean=False, z_q=None):

        with torch.no_grad():
            prior_out = self.prior(img, mean=mean, z_q=z_q)
            encoder_features = prior_out["encoder_features"]
            logits = self.f_comb(encoder_features=encoder_features, decoder_features=prior_out["decoder_features"]).unsqueeze(0)

            for i in tqdm(range(mc_n - 1)):
                prior_out = self.prior(encoder_features, mean=mean, z_q=z_q, skip_encoder=True)
                logits = torch.cat(
                    self.f_comb(encoder_features=encoder_features, decoder_features=prior_out["decoder_features"]).unsqueeze(0), dim=0
                )
            # N x B x H x W x C
            logits = logits.permute(0, 1, 3, 4, 2).float()
            return torch.sigmoid(logits[...,[0]]), logits[...,1:]

    def reconstruct(self, seg, img, mean):
        self.forward(seg, img, mean)
        prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(self, seg, img, lesion_area, skin_area=None, mean=False):
        logits = self.reconstruct(torch.cat((lesion_area, seg), dim=1), img, mean=mean)

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
        link_mat_0, link_mat_1, link_mat = self.log_cumulative(logits[:, 1:])
        y_true = F.one_hot(y_true)
        nll = -torch.log(link_mat_0 * y_true[:, :, [0]] + link_mat_1 * y_true[:, :, [1]] + (link_mat * y_true[:, :, 2:]).sum(-1, keepdim=True)).reshape(
            batch_size, -1, self._num_classes
        )

        xe = F.binary_cross_entropy_with_logits(logits[:, 0], lesion_area, reduction="none").reshape(batch_size, -1)

        lesion_area = torch.reshape(lesion_area, shape=(batch_size, -1, 1))
        skin_area = torch.reshape(skin_area, shape=(batch_size, -1))

        return {
            "mean": (lesion_area * nll).sum() / lesion_area.sum() + (skin_area * xe).sum() / skin_area.sum(),
            "sum": ((lesion_area * nll).sum((1, 2)) + (skin_area * xe).sum(dim=1)).mean(0),
            "mask": skin_area,
        }

    def log_cumulative(self, logits):
        """Convert logits to probability
        https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py

        Args:
            logits (_type_): _description_

        Returns:
            _type_: _description_
        """
        # logits : (N x C)
        # cutpoints : (C x num_cuts)
        logits = logits.unsqueeze(-1)

        # link_mat_0 : (N x C x 1)
        link_mat_0 = torch.sigmoid(-logits)
        # sigmoids : (N x C x num_cuts-1)
        sigmoids = torch.sigmoid(self.cutpoints * self._alpha - logits)

        link_mat_1 = sigmoids[..., [0]] - link_mat_0
        link_mat = torch.cat((sigmoids[..., 1:] - sigmoids[..., :-1], 1 - sigmoids[..., [-1]]), dim=-1)
        
        return link_mat_0, link_mat_1, link_mat

    def kl_divergence(self, seg, img):
        q_dists = self._q_sample["distributions"]
        p_dists = self._p_sample_z_q["distributions"]

        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = kl.kl_divergence(p, q)
            kl_per_instance = kl_per_pixel.sum(dim=(1, 2))
            kl_dict[level] = kl_per_instance.mean()

        return kl_dict

    def sum_loss(
        self,
        seg,
        img,
        lesion_area,
        skin_area=None,
        mean=False,
    ):
        summaries = {}

        # rec loss
        rec_loss = self.rec_loss(seg, img, lesion_area, skin_area, mean)
        summaries["rec_loss_mean"] = rec_loss["mean"]

        # kl loss
        kl_dict = self.kl_divergence(seg, img)
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
