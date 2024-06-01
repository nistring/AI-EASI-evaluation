# Adapted from
# https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch
# https://github.com/google-deepmind/deepmind-research/blob/master/hierarchical_probabilistic_unet
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl

from .model_utils import *
from typing import List, Tuple, Optional


class _HierarchicalCore(nn.Module):
    """A U-Net encoder-decoder with a full encoder and a truncated decoder.

    The truncated decoder is interleaved with the hierarchical latent space and
    has as many levels as there are levels in the hierarchy plus one additional
    level.
    """
    def __init__(
        self,
        latent_dims,
        in_channels,
        channels_per_block,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
    ):
        """Initializes a HierarchicalCore.

        Args:
            latent_dims (List): List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net decoder
                scales that have latents.
            in_channels (int): An integer specifying the number of input channel.
            channels_per_block (List[int]): A list of integers specifying the number of output
                channels for each encoder block.
            down_channels_per_block (List[int], optional): A list of integers specifying the number of
                intermediate channels for each encoder block or None. If None, the
                intermediate channels are chosen equal to channels_per_block. Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function. Defaults to nn.ReLU().
            convs_per_block (int, optional): An integer specifying the number of convolutional layers. Defaults to 3.
            blocks_per_level (int, optional): An integer specifying the number of residual blocks per
                level. Defaults to 3.
        """

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
        """
        Args:
            inputs (torch.Tensor): A tensor of shape (b,c,h,w). When using the module as a prior the
                `inputs` tensor should be a batch of images. When using it as a posterior
                the tensor should be a (batched) concatentation of images and
                segmentations.
            z_q (Optional[List[torch.Tensor]], optional): None or a list of tensors. If not None, z_q provides external latents
                to be used instead of sampling them. This is used to employ posterior
                latents in the prior during training. Therefore, if z_q is not None, the
                value of `mean` is ignored. If z_q is None, either the distributions
                mean is used (in case `mean` for the respective scale is True) or else
                a sample from the distribution is drawn. Defaults to None.
            skip_encoder (Optional[List[torch.Tensor]], optional): A list of tensors that contains
                the output feature map of the encoder. If the skip_encoder is not none, a redundant
                the output feature skips the encoder network and directly processed by the decoder. Defaults to None.
            mean (bool, optional): A boolean or a list of booleans. If a boolean, it specifies whether
                or not to use the distributions' means in ALL latent scales. If a list,
                each bool therein specifies whether or not to use the scale's mean. If
                False, the latents of the scale are sampled.. Defaults to False.

        Returns:
            Tuple[List[torch.Tensor], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
                 A Tuple holding the output feature map of the truncated U-Net
                decoder under key 'decoder_features', a list of the U-Net encoder features
                produced at the end of each encoder scale under key 'encoder_outputs', a
                list of the predicted distributions at each scale under key
                'distributions', a list of the used latents at each scale under the key
                'used_latents'.       
        """
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
    """A module that completes the truncated U-Net decoder.

    Using the output of the HierarchicalCore this module fills in the missing
    decoder levels such that together the two form a symmetric U-Net.
    """
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
    ):
        """Initializes a StichtingDecoder.

        Args:
            latent_dims (List[int]): List of integers specifying the dimensions of the latents at
                each scale. The length of the list indicates the number of U-Net
                decoder scales that have latents.
            in_channels (int): An integer specifying the number of input channel.
            channels_per_block (List[int]): A list of integers specifying the number of output
                channels for each encoder block.
            num_classes (int): An integer specifying the number of segmentation classes.
            down_channels_per_block (List[int], optional): A list of integers specifying the number of
                intermediate channels for each encoder block. If None, the
                intermediate channels are chosen equal to channels_per_block.. Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function. Defaults to nn.ReLU().
            convs_per_block (int, optional): An integer specifying the number of convolutional layers. Defaults to 3.
            blocks_per_level (int, optional): An integer specifying the number of residual blocks per
                level. Defaults to 3.
        """
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
        """Returns the segmentation logits.

        Args:
            encoder_features (List[torch.Tensor]): A list of tensors of shape (b,c_i,h_i,w_i).
            decoder_features (torch.Tensor): A tensor of shape (b,h,w,c).

        Returns:
            torch.Tensor: Logits, i.e. a tensor of shape (b,num_classes,h,w).
        """
        start_level = self.start_level
        for i, decoder in enumerate(self.decoder):
            decoder_features = decoder(decoder_features)
            if i % (self._blocks_per_level + 1) == 0:
                encoder_feature = encoder_features[::-1][start_level]
                decoder_features = torch.cat([decoder_features, encoder_feature], dim=1)
                start_level += 1

        return self.last_layer(decoder_features)


class HierarchicalProbUNet(nn.Module):
    """A Hierarchical Probabilistic U-Net."""
    def __init__(
        self,
        latent_dims=(1, 1, 1, 1),
        in_channels=3,
        channels_per_block=None,
        num_classes=4,
        down_channels_per_block=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
        blocks_per_level=3,
        loss_kwargs=None,
        num_cuts=3,
        weights=None,
    ):
        """Initializes a HierarchicalProbUNet.

        Args:
            latent_dims (tuple, optional): Tuple of integers specifying the dimensions of the latents at
                each scales. The length of the list indicates the number of U-Net
                decoder scales that have latents. Defaults to (1, 1, 1, 1).
            in_channels (int, optional): An integer specifying the number of input channel. Defaults to 3.
            channels_per_block (List[int], optional): A list of integers specifying the number of output
                channels for each encoder block. Defaults to None.
            num_classes (int, optional): An integer specifying the number of segmentation classes. Defaults to 4.
            down_channels_per_block (List[int], optional): A list of integers specifying the number of
                intermediate channels for each encoder block. If None, the
                intermediate channels are chosen equal to channels_per_block. Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function. Defaults to nn.ReLU().
            convs_per_block (int, optional): An integer specifying the number of convolutional layers. Defaults to 3.
            blocks_per_level (int, optional): An integer specifying the number of residual blocks per
                level. Defaults to 3.
            loss_kwargs (Dict, optional): None or dictionary specifying the loss setup. Defaults to None.
            num_cuts (int, optional): An integer specifying the number of threshold values for ordinal regression. Defaults to 3.
            weights (torch.Tensor, optional): A tensor containing weights to compensate class imbalance. Defaults to None.
        """
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
        )

        self.posterior = _HierarchicalCore(
            latent_dims=latent_dims,
            in_channels=in_channels + num_classes,
            channels_per_block=channels_per_block,
            down_channels_per_block=down_channels_per_block,
            activation_fn=activation_fn,
            convs_per_block=convs_per_block,
            blocks_per_level=blocks_per_level,
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
        )

        self._loss_kwargs = loss_kwargs
        self._alpha = nn.Parameter(torch.Tensor([self._loss_kwargs["alpha"]]), requires_grad=False)
        self.cutpoints = nn.Parameter(
            torch.arange(1, num_cuts, dtype=torch.float32, requires_grad=True).expand((num_classes, 1, num_cuts - 1)).clone() / self._alpha
        )
        self._num_cuts = num_cuts

        # Weights for class imbalance
        if weights is None:
            weights = torch.ones((num_classes, num_cuts + 1))
        self._weights = torch.Tensor(weights)

    def _apply(self, fn):
        super(HierarchicalProbUNet, self)._apply(fn)
        # Apply to(device) to member tensor(weights)
        self._weights = fn(self._weights)
        return self

    def forward(self, seg: torch.Tensor, img: torch.Tensor, mean: bool):
        """_summary_

        Args:
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
            img (torch.Tensor): A tensor of shape (b, c, h, w)
            mean (bool): A boolean or a list of booleans. If a boolean, it specifies whether
                or not to use the distributions' means in ALL latent scales. If a list,
                each bool therein specifies whether or not to use the scale's mean. If
                False, the latents of the scale are sampled.

        Returns:
            Output feature maps of the prior and posterior networks.
        """
        _q_sample = self.posterior(inputs=torch.concat([seg, img], dim=1), mean=mean, z_q=[])
        _p_sample_z_q = self.prior(inputs=img, z_q=_q_sample[3])  # used_latents
        return _q_sample, _p_sample_z_q

    @torch.jit.export
    def sample(self, img: torch.Tensor, mc_n: int, mean: bool = False):
        """Sample segmentations from the prior, given an input image.

        Args:
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            mc_n (int): An integer of Monte Carlo sampling trials.
            mean (bool, optional): A boolean or a list of booleans. If a boolean, it specifies whether
                or not to use the distributions' means in ALL latent scales. If a list,
                each bool therein specifies whether or not to use the scale's mean. If
                False, the latents of the scale are sampled. Defaults to False.

        Returns:
            torch.Tensor: A tensor of shape (mc_n, b, c, h, w, num_cut+1).
        """

        with torch.no_grad():
            encoder_features, decoder_features, _, _ = self.prior(inputs=img, mean=mean)
            logits = self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features).unsqueeze(0)

            for i in range(mc_n - 1):
                encoder_features, decoder_features, _, _ = self.prior(inputs=img, mean=mean, skip_encoder=encoder_features)
                logits = torch.cat(
                    (logits, self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features).unsqueeze(0)),
                    dim=0,
                )

            logit_shape = logits.shape
            logits = logits.reshape((-1,) + logit_shape[2:])  # (NB)CHW

            return log_cumulative(self.cutpoints * self._alpha, logits).reshape(logit_shape + (-1,))  # NBCHWc

    def reconstruct(self, seg: torch.Tensor, img: torch.Tensor, mean: bool):
        """Reconstruct a segmentation using the posterior.

        Args:
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            mean (bool): A boolean, specifying whether to sample from the full hierarchy of
                the posterior or use the posterior means at each scale of the hierarchy.

        Returns:
            A segmentation tensor of shape (b, num_classes, h, w) with outputs of the prior and posterior.
        """
        _q_sample, _p_sample_z_q = self.forward(seg, img, mean)
        encoder_features, decoder_features, _, _ = _p_sample_z_q

        return _q_sample, _p_sample_z_q, self.f_comb(encoder_features=encoder_features, decoder_features=decoder_features)

    def dice_loss(self, pred: torch.Tensor, seg: torch.Tensor):
        """Dice loss

        Args:
            pred (torch.Tensor): Predicted class probabilities
            seg (torch.Tensor): True class assignment.

        Returns:
            torch.Tensor: A tensor of shape (b,)
        """
        loss = dice_score(pred, seg) * self._weights  # BCc
        mask = loss != 0
        return (1 - (loss * mask).sum(2) / mask.sum(2)).sum(1).mean() # B

    def ce_loss(self, pred: torch.Tensor, seg: torch.Tensor):
        """Cross-entropy loss

        Args:
            pred (torch.Tensor): Predicted class probabilities
            seg (torch.Tensor): True class assignment.

        Returns:
            torch.Tensor:
        """
        loss = 0.
        pred = torch.log(pred.permute((0, 1, 4, 2, 3)).contiguous().clamp_min(1.0e-7)) # BCcHW
        for i in range(pred.shape[1]):
            loss += F.nll_loss(pred[:, i], seg[:, i], self._weights[i]).mean()

        return loss

    def rec_loss(self, seg: torch.Tensor, img: torch.Tensor, mean: bool = False):
        """Reconstruction loss

        Args:
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            mean (bool, optional): A boolean, specifying whether to sample from the full hierarchy of
                the posterior or use the posterior means at each scale of the hierarchy. Defaults to False.

        Returns:
            Reconstruction loss
        """
        _q_sample, _p_sample_z_q, logits = self.reconstruct(seg, img, mean=mean)
        pred = log_cumulative(self.cutpoints * self._alpha, logits)  # BCHWc
        
        # loss = self.ce_loss(pred, seg)
        loss = self.dice_loss(pred, F.one_hot(seg, num_classes=self._num_cuts + 1))

        return _q_sample, _p_sample_z_q, loss

    def kl_divergence(self, q_dists: torch.Tensor, p_dists: torch.Tensor):
        """Kullback-Leibler divergence between the posterior and the prior.

        Args:
            q_dists (torch.Tensor): A distribution of latent variables of the posterior.
            p_dists (torch.Tensor): A distribution of latent variables of the prior.

        Returns:
            Dict[torch.Tensor]: A dictionary with keys indexing the hierarchy's levels and corresponding
                values holding the KL-term for each level (per batch).
        """
        kl_dict = {}
        for level, (q, p) in enumerate(zip(q_dists, p_dists)):
            kl_per_pixel = kl.kl_divergence(p, q)
            kl_per_instance = kl_per_pixel.sum(dim=(1, 2))
            kl_dict[level] = kl_per_instance.mean()

        return kl_dict

    def sum_loss(self, seg: torch.Tensor, img: torch.Tensor, mean: bool = False):
        """The full training objective, ELBO.

        Args:
            seg (torch.Tensor): A tensor of shape (b, num_classes, h, w).
            img (torch.Tensor): A tensor of shape (b, c, h, w).
            mean (bool, optional): A boolean, specifying whether to sample from the full hierarchy of
                the posterior or use the posterior means at each scale of the hierarchy. Defaults to False.

        Returns:
            A dictionary holding the loss (with key 'loss') and the tensorboard
            summaries (with key 'summaries').
        """
        summaries = {}

        # rec loss
        _q_sample, _p_sample_z_q, rec_loss = self.rec_loss(seg, img, mean)
        summaries["rec_loss"] = rec_loss

        # kl loss
        kl_dict = self.kl_divergence(_q_sample[2], _p_sample_z_q[2])
        kl_sum = torch.stack([kl for _, kl in kl_dict.items()], dim=-1).sum()
        summaries["kl_sum"] = kl_sum
        for level, kl in kl_dict.items():
            summaries["kl_{}".format(level)] = kl

        # ELBO
        loss = rec_loss + self._loss_kwargs["beta"] * kl_sum

        summaries["loss"] = loss
        return dict(supervised_loss=loss, summaries=summaries)
