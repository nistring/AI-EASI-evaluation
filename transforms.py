import torch
from torchvision.transforms import v2

class CustomCrop(torch.nn.Module):
    def __init__(self, size, th=0.125, n=32):
        super().__init__()
        self.size = size
        self.th = th
        self.n = n

    def forward(self, img):
        random_size = min(min(img.shape[1:]), torch.randint(self.size[0], self.size[1]+1, (1,)).item())
        x_high = img.shape[2] - random_size
        y_high = img.shape[1] - random_size
        images = []
        for i in range(self.n): 
            while True:
                x = torch.randint(x_high, (1,)).item()
                y = torch.randint(y_high, (1,)).item()
                crop = img[:, y : y + random_size, x : x + random_size]
                if torch.any(crop > 0):
                    break
            images.append(crop)

        return torch.stack(images)

test_transforms = v2.Compose(
    [
        v2.Resize((256, 256), antialias=True),
    ]
)

roi_test_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=180, translate=(0.5, 0.5)),
        v2.Resize((256, 256), antialias=True),
    ]
)

train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(),
        v2.RandomRotation(180),
        v2.RandomResizedCrop((256, 256), scale=(0.25, 1.0), antialias=True, ratio=(1.0, 1.0)),
        v2.RandomApply([v2.RandomResize(128, 256, antialias=True), v2.Resize((256, 256), antialias=True)]),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
)

roi_train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(),
        v2.RandomAffine(degrees=180, translate=(0.5, 0.5)),
        v2.RandomResizedCrop((256, 256), scale=(0.25, 1.0), antialias=True, ratio=(1.0, 1.0)),
        v2.RandomResize(128, 256, antialias=True),
        v2.Resize((256, 256), antialias=True),
        v2.RandomErasing(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
)


wholebody_train_transforms = v2.Compose(
    [
        v2.Pad(128),
        v2.RandomHorizontalFlip(),
        v2.RandomPerspective(),
        v2.RandomRotation(180),
        CustomCrop((256, 512)),
        v2.Resize((256, 256), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ]
)

wholebody_test_transforms = v2.Compose(
    [
        v2.Pad(128),
        CustomCrop((384, 384)),
        v2.Resize((256, 256), antialias=True)
    ]
)

norm_transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.7040, 0.5334, 0.4461), std=(0.1202, 0.1145, 0.1158)),
    ]
)