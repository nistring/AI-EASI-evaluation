import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from . import geco_utils
from .model import _HierarchicalCore, _StitchingDecoder, HierarchicalProbUNet
from .unet_utils import *


class _MultitaskCore(nn.Module):
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
        super(_MultitaskCore, self).__init__()
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
            if level == 0:
                # classification; no sampling from distributions.
                # Here latent_dim becomes the number of classes for classification task.
                self.in_channels = latent_dim + self._channels_per_block[-2 - level]
                decoder.append(nn.Conv2d(channels, latent_dim, kernel_size=(1, 1), padding=0))
            else:
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

    def forward(self, inputs, grade=None, mean=False, z_q=None, skip_encoder=False):
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
            decoder_features = decoder(decoder_features)
            if type(decoder) == nn.Conv2d:
                if i == 0:  # classification
                    if grade is not None:
                        decoder_features = grade.unsqueeze(-1).unsqueeze(-1)
                    else:
                        decoder_features = F.softmax(decoder_features, dim=1)
                    distributions.append(decoder_features)
                    used_latents.append(decoder_features)
                    self.decoder_output_lo = decoder_features
                else:
                    mu = decoder_features[::, : self._latent_dims[i]]

                    log_sigma = F.softplus(decoder_features[::, self._latent_dims[i] :])

                    norm = Normal(loc=mu, scale=log_sigma)
                    dist = Independent(norm, 1)
                    distributions.append(dist)

                    if z_q is not None:
                        z = z_q[i]
                    elif mean[i]:
                        z = dist.mean
                    else:
                        z = dist.sample()

                    used_latents.append(z)
                    self.decoder_output_lo = torch.concat([z, decoder_features], dim=1)
                i += 1
            if type(decoder) == Resize_up:
                decoder_output_hi = decoder(self.decoder_output_lo)
                decoder_features = torch.concat([decoder_output_hi, encoder_outputs[::-1][j + 1]], dim=1)
                j += 1

        return {
            "decoder_features": decoder_features,
            "encoder_features": encoder_outputs,
            "distributions": distributions,
            "used_latents": used_latents,
        }


class MultiTaskHPU(nn.Module):
    def __init__(
        self,
        latent_dims=(4, 1, 1, 1),  # The first dimension implies the number of class for classification task
        in_channels=3,
        channels_per_block=None,
        num_classes=2,  # For segmentation
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        loss_kwargs=None,
    ):
        super(MultiTaskHPU, self).__init__()

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

        self._loss_kwargs = loss_kwargs

        self.prior = _MultitaskCore(
            latent_dims=latent_dims,
            in_channels=in_channels,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
            name="prior",
        )

        self.posterior = _MultitaskCore(
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

        if self._loss_kwargs["type"] == "geco":
            self._moving_average = geco_utils.MovingAverage(decay=self._loss_kwargs["decay"], differentiable=True)
            self._lagmul = geco_utils.LagrangeMultiplier(rate=self._loss_kwargs["rate"])
        # self._cache = ()

    def forward(self, seg, img, grade):
        inputs = (seg, img)
        # if self._cache != inputs:
        self._q_sample = self.posterior(torch.concat([seg, img], dim=1), grade=grade, mean=False)
        self._q_sample_mean = self.posterior(torch.concat([seg, img], dim=1), grade=grade, mean=True)
        self._p_sample = self.prior(img, mean=False, z_q=None)
        self._p_sample_z_q = self.prior(img, z_q=self._q_sample["used_latents"])
        self._p_sample_z_q_mean = self.prior(img, z_q=self._q_sample_mean["used_latents"])
        # self._cache = inputs
        return

    def sample(self, img, mc_n: int, mean=False, z_q=None):
        prior_out = self.prior(img, mean=mean, z_q=z_q)
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]
        preds = F.softmax(self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features), dim=1)[:, 1]

        for i in range(mc_n - 1):
            prior_out = self.prior(encoder_features, mean=mean, z_q=z_q, skip_encoder=True)
            decoder_features = prior_out["decoder_features"]
            preds = torch.cat(
                (preds, F.softmax(self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features), dim=1)[:, 1]), dim=0
            )

        return preds, prior_out["used_latents"][0].squeeze(-1).squeeze(-1)

    def reconstruct(self, seg, img, grade, mean=False):
        self.forward(seg, img, grade)
        if mean:
            prior_out = self._p_sample_z_q_mean
        else:
            prior_out = self._p_sample_z_q
        encoder_features = prior_out["encoder_features"]
        decoder_features = prior_out["decoder_features"]

        return self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def rec_loss(self, seg, img, grade, mask=None, top_k_percentage=None, deterministic=True):
        reconstruction = self.reconstruct(seg, img, grade, mean=False)
        return geco_utils.ce_loss(reconstruction, seg, mask, top_k_percentage, deterministic)

    def kl_divergence(self, seg, img):
        posterior_out = self._q_sample
        prior_out = self._p_sample_z_q

        q_dists = posterior_out["distributions"]
        p_dists = prior_out["distributions"]

        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            if level == 0:
                kl_dict[level] = self._loss_kwargs["beta"] * F.kl_div(p.log(), q, reduction="batchmean", log_target=False)
            else:
                kl_per_pixel = kl.kl_divergence(p, q)  # p <-> q in pytorch
                kl_per_instance = kl_per_pixel.sum(axis=[1, 2])
                kl_dict[level] = kl_per_instance.mean()
        return kl_dict

    def sum_loss(self, seg, img, grade, mask=None):
        summaries = {}
        top_k_percentage = self._loss_kwargs["top_k_percentage"]
        deterministic = self._loss_kwargs["deterministic_top_k"]
        rec_loss = self.rec_loss(seg, img, grade, mask, top_k_percentage, deterministic)

        kl_dict = self.kl_divergence(seg, img)
        kl_sum = torch.stack([kl for _, kl in kl_dict.items()], dim=-1).sum()

        summaries["rec_loss_mean"] = rec_loss["mean"]
        summaries["rec_loss_sum"] = rec_loss["sum"]
        summaries["kl_sum"] = kl_sum

        for level, kl in kl_dict.items():
            summaries["kl_{}".format(level)] = kl

        if self._loss_kwargs["type"] == "elbo":
            loss = rec_loss["sum"] + self._loss_kwargs["beta"] * kl_sum
            summaries["elbo_loss"] = loss

        elif self._loss_kwargs["type"] == "geco":
            ma_rec_loss = self._moving_average(rec_loss["sum"])
            mask_sum_per_instance = rec_loss["mask"].sum(dim=-1)
            num_valid_pixels = mask_sum_per_instance.mean()
            reconstruction_threshold = self._loss_kwargs["kappa"] * num_valid_pixels

            rec_constraint = ma_rec_loss - reconstruction_threshold
            lagmul = self._lagmul()
            loss = lagmul * rec_constraint + kl_sum

            summaries["geco_loss"] = loss
            summaries["ma_rec_loss_mean"] = ma_rec_loss / num_valid_pixels
            summaries["num_valid_pixels"] = num_valid_pixels
            summaries["lagmul"] = lagmul
        else:
            raise NotImplementedError("Loss type {} not implemeted!".format(self._loss_kwargs["type"]))

        return dict(supervised_loss=loss, summaries=summaries)
