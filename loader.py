from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools

image_size = 256

train_transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.Rotate(limit=180),
        A.GaussNoise(p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=15, p=0.2), A.MedianBlur(blur_limit=15, p=0.1), A.Blur(blur_limit=15, p=0.1)], p=0.5),
        A.OneOf([A.GridDistortion(), A.ElasticTransform(), A.OpticalDistortion()], p=0.5),
        A.Perspective(),
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.RandomResizedCrop(image_size, image_size, scale=(0.5, 1)),
        A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [A.Resize(image_size, image_size), A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243)), ToTensorV2()]
)

class CustomDataset(Dataset):
    def __init__(self, root: str, transforms=test_transforms):
        self.root = root
        self.imgs = sorted(os.listdir(root))
        self.transforms = transforms

    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        img = cv2.imread(os.path.join(self.root, file_name))
        sample = {
            "transformed":self.transforms(image=cv2.resize(img, (512, 512)))['image'],
            'ori_img':img,
            'file_name':file_name.split('.')[0] + '.png'
        }
        return sample

    def __len__(self):
        return len(self.imgs)


class AtopyDataset(Dataset):
    def __init__(self, imgs, masks, classes, grades, transforms):
        self.transforms = transforms
        self.imgs = imgs
        self.masks = masks
        self.classes = classes
        self.grades = grades
        self.labellers = os.listdir("data/train/labels")

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.resize(img, (512, 512))

        masks = []
        for labeller in self.labellers:
            mask = cv2.imread(os.path.join("data/train/labels", labeller, self.masks[idx]), 0)
            mask = cv2.resize(mask, (512, 512))
            mask[mask != 0] = 1
            masks.append(mask)

        transformed = self.transforms(image=img, masks=masks)
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[self.grades[idx]] = 1.0
        masks = np.stack(transformed["masks"], axis=0)
        masks = np.stack([1 - masks, masks], axis=1) # background / target
        return masks, transformed["image"], one_hot

    def __len__(self):
        return len(self.imgs)


def get_file_list(phase):
    labellers = os.listdir("data/train/labels")

    imgs = []
    masks = []
    classes = []
    grades = []
    for g, grade in enumerate(["Grade0", "Grade1", "Grade2", "Grade3"]):
        intersection = []
        for labeller in labellers:
            intersection.append(set(map(lambda x: x.split(".")[0], os.listdir(os.path.join("data/train/labels", labeller, phase, grade)))))
        intersection.append(set(map(lambda x: x.split(".")[0], os.listdir(os.path.join("data/train/images", phase, grade)))))
        intersection = sorted(list(functools.reduce(lambda a, b: a & b, intersection)))

        imgs.extend(list(map(lambda x: os.path.join("data/train/images", phase, grade, x + ".jpg"), intersection)))
        masks.extend(list(map(lambda x: os.path.join(phase, grade, x + ".png"), intersection)))
        classes.extend([int(grade[-1])] * len(intersection))
        grades.extend([g] * len(intersection))
        print(phase, grade, len(intersection))

    return len(imgs), np.array(imgs), np.array(masks), np.array(classes), np.array(grades)


def get_dataset(phase, split_ratio=0.2):

    total_num_samples, imgs, masks, classes, grades = get_file_list(phase)

    if phase == "Atopy_Segment_Train":
        val_idx = random.sample(range(total_num_samples), int(total_num_samples * split_ratio))
        train_idx = [x for x in range(total_num_samples) if x not in val_idx]

        return AtopyDataset(imgs[train_idx], masks[train_idx], classes[train_idx], grades[train_idx], train_transforms), AtopyDataset(
            imgs[val_idx], masks[val_idx], classes[val_idx], grades[val_idx], test_transforms
        )
    else:
        return AtopyDataset(imgs, masks, classes, grades, test_transforms)


if __name__ == "__main__":
    # for phase in ["Atopy_Segment_Train", "Atopy_Segment_Test", "Atopy_Segment_Extra"]:
    #     get_file_list(phase)

    cols = 5
    total_num_samples, imgs, masks, classes, grades = get_file_list("Atopy_Segment_Train")
    dataset = AtopyDataset(imgs, masks, classes, grades, train_transforms)
    train_aug = A.Compose([t for t in train_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])
    test_aug = A.Compose([t for t in test_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])

    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(64, 32))
    for i in range(cols):
        idx = random.randint(0, len(dataset))

        img = cv2.imread(dataset.imgs[idx])
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = []
        for labeller in ['동현', '재성']:
            mask = cv2.imread(os.path.join("data/train/labels", labeller, dataset.masks[idx]), 0)
            mask = cv2.resize(mask, (512, 512))
            mask[mask != 0] = 1
            masks.append(mask)

        before = test_aug(image=img, masks=masks)
        mask1, mask2 = before["masks"]
        mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
        mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
        before = np.concatenate([before['image'], mask1, mask2], axis=0)

        after = train_aug(image=img, masks=masks)
        mask1, mask2 = after["masks"]
        mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
        mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
        after = np.concatenate([after['image'], mask1, mask2], axis=0)

        img = np.concatenate([before, after], axis=1)

        ax[i].imshow(img)
        ax[i].set_axis_off()
        ax[i].set_title(f"Image {i}", fontsize=32)

    plt.tight_layout()
    fig.savefig("augmentation.png")

    """
    Adopted from https://kozodoi.me/blog/20210308/compute-image-stats
    """
    total_num_samples, imgs, masks, classes, grades = get_file_list("Atopy_Segment_Train")
    ####### COMPUTE MEAN / STD
    test_transforms = A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=0, std=1), ToTensorV2()])
    dataset = AtopyDataset(imgs, masks, classes, grades, test_transforms)
    image_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for masks, inputs, _ in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])

    ####### FINAL CALCULATIONS

    # pixel count
    count = len(dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
