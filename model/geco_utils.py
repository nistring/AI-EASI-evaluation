import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is a part of implementation of GECO(Generalized ELBO with Constrained Optimization) algorithm
from "Taming VAEs"[https://arxiv.org/pdf/1810.00597.pdf]
"""


class MovingAverage(nn.Module):
    """
    Refer to https://github.com/deepmind/sonnet/blob/v1/sonnet/python/modules/moving_average.py
    """

    def __init__(self, decay, local=True, differentiable=True, name="moving_average"):
        super(MovingAverage, self).__init__()
        self._differentiable = differentiable
        if decay < 0.0 or decay > 1.0:
            raise ValueError("Decay must be a float in the [0, 1] range, " "but is {}.".format(decay))
        self._decay = decay
        self._initialized = False

    def forward(self, inputs):
        if not self._differentiable:
            inputs = inputs.detech()
        if not self._initialized:
            self._moving_average = inputs
            self._initialized = True
        else:
            self._moving_average = (1 - self._decay) * inputs + self._decay * self._moving_average

        return inputs + (self._moving_average - inputs).detach()


def _sample_gumbel(shape, eps=1e-20):
    return -torch.log(-torch.log(shape.uniform_(0, 1) + eps) + eps)


def _topk_mask(score, k):
    device = score.get_device()
    _, indices = torch.topk(score, k)
    k = torch.ones(k).to(device)
    z = torch.zeros(torch.squeeze(score).shape).to(device)
    return z.scatter_(0, indices, k)


def ce_loss(logits, labels, mask=None, top_k_percentage=None, deterministic=False):
    """Computes the cross-entropy loss.
    Optionally a mask and a top-k percentage for the used pixels can be specified.
    The top-k mask can be produced deterministically or sampled.
    Args:
      logits: A tensor of shape (b,h,w,num_classes)
      labels: A tensor of shape (b,h,w,num_classes)
      mask: None or a tensor of shape (b,h,w).
      top_k_percentage: None or a float in (0.,1.]. If None, a standard
        cross-entropy loss is calculated.
      deterministic: A Boolean indicating whether or not to produce the
        prospective top-k mask deterministically.
    Returns:
      A dictionary holding the mean and the pixelwise sum of the loss for the
      batch as well as the employed loss mask.
    """
    device = logits.get_device()
    logits = logits.permute(0, 2, 3, 1)
    labels = labels.permute(0, 2, 3, 1)

    num_classes = list(logits.shape)[-1]
    y_flat = torch.reshape(logits, (-1, num_classes))
    t_flat = torch.reshape(labels, (-1, num_classes))
    if mask is None:
        mask = torch.ones(
            list(t_flat.shape)[0],
        ).to(device)
    else:
        assert list(mask.shape)[:3] == list(labels.shape)[:3], "The loss mask shape differs from the target shape: {} vs. {}.".format(
            list(mask.shape), list(labels.shape)[:3]
        )
        mask = torch.reshape(mask, (-1,))

    n_pixels_in_batch = list(y_flat.shape)[0]
    xe = F.cross_entropy(y_flat, torch.argmax(t_flat, dim=1), reduction="none")
    if top_k_percentage is not None:
        assert 0.0 < top_k_percentage <= 1.0
        k_pixels = torch.floor(n_pixels_in_batch * torch.Tensor([top_k_percentage])).type(torch.int32).item()

        stopgrad_xe = xe.detach()
        norm_xe = stopgrad_xe / stopgrad_xe.sum()

        if deterministic:
            score = torch.log(norm_xe)
        else:
            score = torch.log(norm_xe) + _sample_gumbel(norm_xe)

        score = score + torch.log(mask)
        top_k_mask = _topk_mask(score, k_pixels)
        mask = mask * top_k_mask

    batch_size = list(labels.shape)[0]
    xe = torch.reshape(xe, shape=(batch_size, -1))
    mask = torch.reshape(mask, shape=(batch_size, -1))
    ce_sum_per_instance = (mask * xe).sum(dim=1)
    ce_sum = ce_sum_per_instance.mean(dim=0)
    ce_mean = (mask * xe).sum() / mask.sum()

    return {"mean": ce_mean, "sum": ce_sum, "mask": mask}