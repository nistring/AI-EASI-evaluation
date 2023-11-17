import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from .unet_utils import *
from . import geco_utils


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
            self.in_channels = (2 if level == 0 else 3) * latent_dim + self._channels_per_block[-2 - level]
            decoder.append(nn.Conv2d(channels, (1 if level == 0 else 2) * latent_dim, kernel_size=(1, 1), padding=0))

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
        distributions = []
        used_latents = []
        if isinstance(mean, bool):
            mean = [mean] * self.num_latent_levels
        i = 0
        j = 0

        for decoder in self.decoder:
            if type(decoder) == nn.Conv2d:
                decoder_features = decoder(decoder_features)
                
                if i == 0: # grade
                    z = decoder_features
                else:
                    mu = decoder_features[:, :self._latent_dims[i]]

                    log_sigma = F.softplus(decoder_features[:, self._latent_dims[i]:])

                    norm = Normal(loc=mu, scale=log_sigma)
                    dist = Independent(norm, 1)
                    distributions.append(dist)

                    z = dist.sample()

                    if mean[i]:
                        z = dist.mean

                if z_q is not None:
                    if i < len(z_q):
                        z = z_q[i]

                decoder_output_lo = torch.concat([z, decoder_features], dim=1)
                    
                used_latents.append(z)
                i += 1
            elif type(decoder) == Resize_up:
                decoder_output_hi = decoder(decoder_output_lo)
                decoder_features = torch.concat([decoder_output_hi, encoder_outputs[::-1][j + 1]], dim=1)
                j += 1
            else:
                decoder_features = decoder(decoder_features)

        return {
            "decoder_features": decoder_features,
            "encoder_features": encoder_outputs,
            "distributions": distributions,
            "used_latents": used_latents,
        }


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
        num_classes=2,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        loss_kwargs=None,
        num_grades=(4,4), # (character, severity scores)
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
            in_channels=in_channels + num_classes,
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
            num_classes=num_classes,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="f_comb",
        )

        self._loss_kwargs = loss_kwargs
        # self._moving_average = geco_utils.MovingAverage(decay=self._loss_kwargs["decay"], differentiable=True)

        if self._loss_kwargs["type"] == "geco":
            """
            Refer to https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/optimization_constraints.py
            """
            self._lagmul = nn.Parameter(torch.Tensor([1.0]))
            self._lagmul.register_hook(lambda grad: -grad * self._loss_kwargs["rate"])
            self._lagmul_grade = nn.Parameter(torch.Tensor([1.0]))
            self._lagmul_grade.register_hook(lambda grad: -grad * self._loss_kwargs["rate"])
            self.softplus = nn.Softplus()

        self.cutpoints = nn.Parameter((torch.arange(num_grades[1])).expand(num_grades) - num_grades[1]/2)

    def forward(self, seg, img, grade):
        self._q_sample = self.posterior(torch.concat([seg, img], dim=1))
        self._p_sample_z_q = self.prior(img, z_q=self._q_sample["used_latents"])
        return

    def sample(self, img, mc_n: int, mean=False, z_q=None):
        prior_out = self.prior(img, mean, z_q)
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        # preds = torch.argmax(self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features), dim=1)
        preds = F.softmax(self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features), dim=1)[:, 1]
        grades = self.log_cumulative(prior_out["used_latents"][0]).argmax(-1)

        for i in range(mc_n - 1):
            prior_out = self.prior(encoder_features, mean, z_q, skip_encoder=True)
            decoder_features = prior_out["decoder_features"]
            preds = torch.cat(
                (preds, torch.argmax(self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features), dim=1)), dim=0
            )
            grades = torch.cat((grades, self.log_cumulative(prior_out["used_latents"][0]).argmax(-1)))

        return preds, grades

    def reconstruct(self, seg, img, grade, mean=False):
        self.forward(seg, img, grade)
        prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        return self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(self, seg, img, grade=None, mask=None, top_k_percentage=None, deterministic=True):
        reconstruction = self.reconstruct(seg, img, grade, mean=False)
        return geco_utils.ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def log_cumulative(self, logits):
        """Convert logits to probability
        https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py

        Args:
            logits (_type_): _description_

        Returns:
            _type_: _description_
        """
        # logits : (B x C x 1 x 1)
        # cutpoints : (C x num_cuts)
        sigmoids = torch.sigmoid(self.cutpoints - logits.squeeze(-1))
        # sigmoids : (B x C x num_cuts)
        link_mat = torch.cat((
                sigmoids[..., [0]],
                sigmoids[..., 1:] - sigmoids[..., :-1],
                1 - sigmoids[..., [-1]]
            ),
            dim=-1
        )
        return link_mat

    def cls_loss(self, y_true):
        
        q_grade = self._q_sample["used_latents"][0]
        p_grade = self._p_sample_z_q["used_latents"][0]
        loss = dict(mean=0., sum=0.)
        for grade in (q_grade, p_grade):
            # y_true : (B x C)
            # y_pred : (B x C x num_cuts)
            y_pred = self.log_cumulative(grade)
            eps = 1e-15
            likelihoods = torch.clamp(torch.gather(y_pred, 2, y_true.unsqueeze(-1)), eps, 1 - eps)
            neg_log_likelihood = -torch.log(likelihoods)
            # neg_log_likelihood : (B x C x 1)
            loss["mean"] += neg_log_likelihood.mean()
            loss["sum"] += neg_log_likelihood.sum((1,2)).mean()

        return loss

    def kl_divergence(self, seg, img):
        q_dists = self._q_sample["distributions"]
        p_dists = self._p_sample_z_q["distributions"]

        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = kl.kl_divergence(p, q)
            kl_per_instance = kl_per_pixel.sum(dim=(1, 2))
            kl_dict[level] = kl_per_instance.mean()

        return kl_dict

    def sum_loss(self, seg, img, grade=None, mask=None):
        summaries = {}
        top_k_percentage = self._loss_kwargs["top_k_percentage"]
        deterministic = self._loss_kwargs["deterministic_top_k"]

        # rec loss
        rec_loss = self.rec_loss(seg, img, grade, mask, top_k_percentage, deterministic)
        summaries["rec_loss_mean"] = rec_loss["mean"]

        # kl loss
        kl_dict = self.kl_divergence(seg, img)
        kl_sum = torch.stack([kl for _, kl in kl_dict.items()], dim=-1).sum()
        summaries["kl_sum"] = kl_sum
        for level, kl in kl_dict.items():
            summaries["kl_{}".format(level)] = kl

        # classification loss
        if grade is not None:
            cls_loss = self.cls_loss(grade)
            summaries["cls_loss_mean"] = cls_loss["mean"]

        mask_sum_per_instance = rec_loss["mask"].sum(dim=-1)
        num_valid_pixels = mask_sum_per_instance.mean()

        # ELBO
        if self._loss_kwargs["type"] == "elbo":
            loss = rec_loss["sum"] + self._loss_kwargs["beta"] * kl_sum

            if grade is not None:
                loss += self._loss_kwargs["alpha"] * cls_loss["sum"]
            
        # GECO
        elif self._loss_kwargs["type"] == "geco":
            reconstruction_threshold = self._loss_kwargs["kappa"] * num_valid_pixels
            rec_constraint = rec_loss["sum"] - reconstruction_threshold
            lagmul = self.softplus(self._lagmul) ** 2
            loss = lagmul * rec_constraint + kl_sum

            if grade is not None:
                classification_threshold = self._loss_kwargs["kappa_grade"] * grade.shape[-1]
                cls_constraint = self._loss_kwargs["alpha"] * (cls_loss["sum"] - classification_threshold)
                lagmul_grade = self.softplus(self._lagmul_grade) ** 2 
                loss += lagmul_grade * cls_constraint

                summaries["lagmul_grade"] = self._lagmul_grade

            summaries["lagmul"] = self._lagmul
        else:
            raise NotImplementedError("Loss type {} not implemeted!".format(self._loss_kwargs["type"]))

        summaries["loss"] = loss
        return dict(supervised_loss=loss, summaries=summaries)
