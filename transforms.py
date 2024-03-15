import torch
from torchvision.transforms import v2

class CustomCrop(torch.nn.Module):
    def __init__(self, size = (256, 256)):
        super().__init__()
        self.w, self.h = size

    def forward(self, img):
        x_high = max(img.shape[2] - self.w, 1)
        y_high = max(img.shape[1] - self.h, 1)
        while True:
            x = torch.randint(x_high, (1,)).item()
            y = torch.randint(y_high, (1,)).item()
            crop = img[:, y : min(y + self.h, img.shape[1]), x : min(x + self.w, img.shape[2])]
            if torch.any(crop): # If the cropped image contains at least one pixel of foreground image.
                break
        crop = v2.functional.resize(crop, (self.h, self.w), antialias=True)
        return crop

class DownScale(torch.nn.Module):
    def __init__(self, scale_min=0.25, scale_max=1.):
        super().__init__()
        assert (scale_max <= 1.) and (scale_min > 0.) and (scale_min <= scale_max)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, data):
        scale = torch.rand(1).item() * (self.scale_max - self.scale_min) + self.scale_min
        if isinstance(data, tuple):
            img = data[0]
        else:
            img = data
        size = img.shape[1:]
        dsize = [round(size[0] * scale), round(size[1] * scale)]
        img = v2.functional.resize(img, dsize, antialias=True)
        img = v2.functional.resize(img, size, antialias=True)
        if isinstance(data, tuple):
            return img, data[1]
        else:
            return img

test_transforms = v2.Compose([
    v2.Resize((256, 256), antialias=True),
    # v2.ToDtype(torch.float32, scale=True),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204)),
])


train_transforms = v2.Compose([
    v2.RandomVerticalFlip(),
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(),
    v2.RandomResizedCrop((256, 256), (0.8, 1.0), antialias=True, ratio=(1.0, 1.0)),
    v2.RandomApply([DownScale()]),
    v2.ColorJitter(brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05)),
    # v2.ToDtype(torch.float32, scale=True),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204)),
])


wholebody_transforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomPerspective(),
    CustomCrop((256, 256)),
    v2.RandomApply([DownScale()]),
    v2.ColorJitter(brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05)),
    # v2.ToDtype(torch.float32, scale=True),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204)),
])