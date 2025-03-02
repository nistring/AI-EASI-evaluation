# Adapted from
# https://github.com/Zerkoar/hierarchical_probabilistic_unet_pytorch
# https://github.com/google-deepmind/deepmind-research/blob/master/hierarchical_probabilistic_unet
import torch.nn as nn
import torch


def init_weights_orthogonal_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight, gain=1.0)
        nn.init.trunc_normal_(m.bias, mean=0, std=0.001)


def dice_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7) -> torch.Tensor:
    """Calculates dice score.

    Args:
        output (torch.Tensor):
        target (torch.Tensor):
        smooth (float, optional): Defaults to 0.0.
        eps (float, optional): Defaults to 1e-7.

    Returns:
        torch.Tensor: dice score
    """
    assert output.shape == target.shape
    dsize = output.shape[:2] + (-1, output.shape[-1])
    output = output.reshape(dsize)  # N x C x _ x num_cuts+1
    target = target.reshape(dsize)  # N x C x _ x num_cuts+1

    return (2.0 * (output * target).sum(2) + smooth) / ((output + target).sum(2) + smooth).clamp_min(eps)  # N x C x num_cuts+1


def log_cumulative(cutpoints: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probability
    https://github.com/EthanRosenthal/spacecutter/blob/master/spacecutter/models.py

    Args:
        cutpoints (torch.Tensor): C x 1 x num_cuts-1
        logits (torch.Tensor): N x C x ...

    Returns:
        torch.Tensor: _description_
    """
    logit_shape = logits.shape
    logits = logits.reshape((logit_shape[0], logit_shape[1], -1, 1))  # N x C x _ x 1
    p_0 = torch.sigmoid(-logits)  # N x C x _ x 1
    sigmoids = torch.sigmoid(cutpoints - logits)  # N x C x _ x num_cuts-1

    return torch.cat((p_0, sigmoids[..., 0:1] - p_0, sigmoids[..., 1:] - sigmoids[..., :-1], 1 - sigmoids[..., -1:]), dim=-1).reshape(
        logit_shape + (-1,)
    )  # N x C x ... x num_cuts + 1


class Res_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_down_channels=None,
        activation_fn=nn.ReLU(),
        convs_per_block=3,
    ):
        """A pre-activated residual block.

        Args:
            in_channels (int): An integer specifying the number of input channels.
            out_channels (int): An integer specifying the number of output channels.
            n_down_channels (int, optional): An integer specifying the number of intermediate channels.. Defaults to None.
            activation_fn (torch.nn.Module, optional): A callable activation function.. Defaults to nn.ReLU().
            convs_per_block (int, optional): An Integer specifying the number of convolutional layers.. Defaults to 3.
        """
        #  input_features: A tensor of shape (b, c, h, w).
        super(Res_block, self).__init__()
        self.n_down_channels = n_down_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip = nn.Identity()

        if self.n_down_channels is None:
            self.n_down_channels = self.out_channels

        layers = []
        layers.append(activation_fn)
        layers.append(nn.Conv2d(self.in_channels, self.n_down_channels, kernel_size=(3, 3), padding=1))
        layers.append(activation_fn)

        for c in range(convs_per_block - 1):
            layers.append(nn.Conv2d(self.n_down_channels, self.n_down_channels, kernel_size=(3, 3), padding=1))
            if c < convs_per_block - 2:
                layers.append(activation_fn)

        if self.in_channels != self.out_channels:
            self.skip = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), padding=0)
            self.skip.apply(init_weights_orthogonal_normal)

        if self.n_down_channels != self.out_channels:
            layers.append(nn.Conv2d(self.n_down_channels, self.out_channels, kernel_size=(1, 1), padding=0))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights_orthogonal_normal)

    def forward(self, input_features: torch.Tensor):
        if self.in_channels != self.out_channels:
            skip = self.skip(input_features)
        else:
            skip = input_features
        return skip + self.layers(input_features)


class Resize_up(nn.Module):
    def __init__(self, scale=2):
        """Nearest neighbor rescaling-operation for the input features.

        Args:
            scale (int, optional): An integer specifying the scaling factor.. Defaults to 2.
        """
        super(Resize_up, self).__init__()
        assert scale >= 1
        self.scale = scale
        self.up = nn.Upsample(scale_factor=self.scale, mode="bilinear", align_corners=True)

    def forward(self, input_features: torch.Tensor):
        return self.up(input_features)


class Resize_down(nn.Module):
    def __init__(self, scale=2):
        """Average pooling rescaling-operation for the input features.

        Args:
            scale (int, optional): An integer specifying the scaling factor. Defaults to 2.
        """
        super(Resize_down, self).__init__()
        assert scale >= 1
        self.scale = scale
        self.down = nn.AvgPool2d((self.scale, self.scale), stride=(self.scale, self.scale), padding=0)

    def forward(self, input_features: torch.Tensor):
        return self.down(input_features)
