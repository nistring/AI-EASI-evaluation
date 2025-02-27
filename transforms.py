import torch
from torchvision.transforms import v2

class CustomCrop(torch.nn.Module):
    def __init__(self, size, th=0.125, n=10):
        """A custom transformation module that selectively crops nonzero area.

        Args:
            size (Tuple[int, int]): Size of the patch.
            th (float, optional): The threshold value of minimum proportion of nonzero area. Defaults to 0.125.
            n (int, optional): Number of patches to be extracted from a single image. Defaults to 10, which is the same copy number of a roi image.
        """
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
                x = torch.randint(0, x_high, (1,)).item()
                y = torch.randint(0, y_high, (1,)).item()
                crop = img[:, y : y + random_size, x : x + random_size]
                if (crop > 0).sum().item() / crop.numel() > self.th:
                    break
            images.append(crop)

        return torch.stack(images)

# ROI/whole-body test
test_transforms = v2.Compose(
    [
        v2.Resize((256, 256), antialias=True),
    ]
)

# ROI tuning(val) for training whole-body model
roi_test_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=180, translate=(0.5, 0.5)),
        v2.Resize((256, 256), antialias=True),
    ]
)

# ROI train
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

# ROI/whole-body train
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

# whole-body train
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

# whole-body tuning
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