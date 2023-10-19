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
from patchify import patchify
from sklearn.metrics import cohen_kappa_score
import pandas as pd

image_size = 256

train_transforms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(p=0.5),
        A.OneOf([A.MotionBlur(blur_limit=15, p=0.2), A.MedianBlur(blur_limit=15, p=0.1), A.Blur(blur_limit=15, p=0.1)], p=0.5),
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.Perspective(),
        A.OneOf(
            [
                A.GridDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            ],
            p=0.5,
        ),
        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [A.Resize(image_size, image_size), A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243)), ToTensorV2()]
)


class CustomDataset(Dataset):
    def __init__(self, root: str):
        self.root = root
        self.imgs = sorted(os.listdir(root))
        self.patch_size = 578 * 2

    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        ori_img = cv2.imread(os.path.join(self.root, file_name))
        re_h = int(ori_img.shape[0] // (self.patch_size / 2) * 128)
        re_w = int(ori_img.shape[1] // (self.patch_size / 2) * 128)
        ori_img = cv2.resize(ori_img, (re_w, re_h))

        mask = (ori_img[:, :, 0] >= 250) & (ori_img[:, :, 1] >= 250) & (ori_img[:, :, 2] >= 250)
        mask = np.stack((mask, mask, mask), axis=-1)
        ori_img[mask] = 0

        transforms = A.Compose([A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243))])
        img = transforms(image=ori_img)["image"]

        patches = patchify(np.transpose(img, (2, 0, 1)), (3, 256, 256), step=128)

        sample = {
            "patches": patches,
            "ori_img": ori_img,
            "file_name": file_name.split(".")[0] + ".png",
            "mask": mask
        }
        return sample

    def __len__(self):
        return len(self.imgs)


class AtopyDataset(Dataset):
    def __init__(self, imgs, masks, classes, transforms):
        self.transforms = transforms
        self.imgs = imgs
        self.masks = masks
        self.grades = pd.read_pickle("data/train/classes/Mean_class.pkl")
        self.classes = classes
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
        masks = np.stack(transformed["masks"], axis=0)
        masks = np.stack([1 - masks, masks], axis=1)  # background / target
        grade = self.grades.loc[int(os.path.basename(self.imgs[idx]).split('.')[0])].to_numpy().astype(np.float32)

        return masks, transformed["image"], grade, img

    def __len__(self):
        return len(self.imgs)


def get_file_list(phase):
    labellers = os.listdir("data/train/labels")

    imgs = []
    masks = []
    classes = []
    for g, grade in enumerate(["Grade0", "Grade1", "Grade2", "Grade3"]):
        intersection = []
        for labeller in labellers:
            intersection.append(set(map(lambda x: x.split(".")[0], os.listdir(os.path.join("data/train/labels", labeller, phase, grade)))))
        intersection.append(set(map(lambda x: x.split(".")[0], os.listdir(os.path.join("data/train/images", phase, grade)))))
        intersection = sorted(list(functools.reduce(lambda a, b: a & b, intersection)))

        imgs.extend(list(map(lambda x: os.path.join("data/train/images", phase, grade, x + ".jpg"), intersection)))
        masks.extend(list(map(lambda x: os.path.join(phase, grade, x + ".png"), intersection)))
        classes.extend([int(grade[-1])] * len(intersection))
        print(phase, grade, len(intersection))

    return len(imgs), np.array(imgs), np.array(masks), np.array(classes)


def get_dataset(phase, split_ratio=0.2):

    total_num_samples, imgs, masks, classes = get_file_list(phase)

    if phase == "Atopy_Segment_Train":
        val_idx = random.sample(range(total_num_samples), int(total_num_samples * split_ratio))
        train_idx = [x for x in range(total_num_samples) if x not in val_idx]

        return AtopyDataset(imgs[train_idx], masks[train_idx], classes[train_idx], train_transforms), AtopyDataset(
            imgs[val_idx], masks[val_idx], classes[val_idx], test_transforms
        )
    else:
        return AtopyDataset(imgs, masks, classes, test_transforms)


if __name__ == "__main__":
    # for phase in ["Atopy_Segment_Train", "Atopy_Segment_Test", "Atopy_Segment_Extra"]:
    #     get_file_list(phase)

    cols = 10
    total_num_samples, imgs, masks, classes = get_file_list("Atopy_Segment_Train")
    dataset = AtopyDataset(imgs, masks, classes, train_transforms)
    train_aug = A.Compose([t for t in train_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])
    test_aug = A.Compose([t for t in test_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])

    mean_h, mean_w = 0, 0
    for i in tqdm(range(len(dataset))):
        img = cv2.imread(dataset.imgs[i])
        h, w = img.shape[:2]
        mean_h += h
        mean_w += w
    mean_h /= len(dataset)
    mean_w /= len(dataset)
    print(f"mean size : ({mean_h}, {mean_w})")

    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(64, 16))
    for i in range(cols):
        idx = random.randint(0, len(dataset)-1)

        img = cv2.imread(dataset.imgs[idx])
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = []
        for labeller in ["동현", "재성"]:
            mask = cv2.imread(os.path.join("data/train/labels", labeller, dataset.masks[idx]), 0)
            mask = cv2.resize(mask, (512, 512))
            mask[mask != 0] = 1
            masks.append(mask)

        before = test_aug(image=img, masks=masks)
        mask1, mask2 = before["masks"]
        mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
        mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
        before = np.concatenate([before["image"], mask1, mask2], axis=0)

        after = train_aug(image=img, masks=masks)
        mask1, mask2 = after["masks"]
        mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
        mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
        after = np.concatenate([after["image"], mask1, mask2], axis=0)

        img = np.concatenate([before, after], axis=1)

        ax[i].imshow(img)
        ax[i].set_axis_off()
        ax[i].set_title(f"Image {i}", fontsize=32)

    plt.tight_layout()
    fig.savefig("augmentation.png")

    """
    Adopted from https://kozodoi.me/blog/20210308/compute-image-stats
    """
    total_num_samples, imgs, masks, classes = get_file_list("Atopy_Segment_Train")
    ####### COMPUTE MEAN / STD
    test_transforms = A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=0, std=1), ToTensorV2()])
    dataset = AtopyDataset(imgs, masks, classes, test_transforms)
    image_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    cohens_k = .0

    # loop through images
    for mask, inputs, grade, img in tqdm(image_loader):
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])
        mask = mask.numpy()
        for i in range(mask.shape[0]):
            cohens_k += cohen_kappa_score(mask[i, 0].reshape(-1), mask[i, 1].reshape(-1))

    ####### FINAL CALCULATIONS

    # pixel count
    count = len(dataset) * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)
    total_cohen = cohens_k / len(dataset)

    # output
    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))
    print("cohen's k:  " + str(total_cohen))
