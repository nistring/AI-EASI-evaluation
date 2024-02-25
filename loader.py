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
from patchify import patchify
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

image_size = 256

train_transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.GaussNoise(),
        A.Blur(),
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.Perspective(),
        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, scale_limit=0.2, value=0.0),
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [A.Resize(image_size, image_size), A.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204)), ToTensorV2()]
)


class CustomDataset(Dataset):
    def __init__(self, root: str, step: int):
        self.root = root
        self.imgs = sorted(os.listdir(root))
        self.patch_size = 645
        assert step >= 256 / 2
        self.step = step

    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        ori_img = cv2.imread(os.path.join(self.root, file_name))
        ori_img = cv2.resize(int(ori_img.shape[1] / self.patch_size * 256), (re_w, int(ori_img.shape[0] / self.patch_size * 256)))
        ori_img = np.pad(
            ori_img,
            (
                (256 - self.step, ori_img.shape[0] // self.step * self.step - ori_img.shape[0] + 256),
                (256 - self.step, ori_img.shape[1] // self.step * self.step - ori_img.shape[1] + 256),
            ),
        )

        mask = ori_img.sum(2) == 0
        mask = np.stack((mask, mask, mask), axis=-1)
        ori_img[mask] = 0

        transforms = A.Compose([A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243))])
        img = transforms(image=ori_img)["image"]

        patches = patchify(np.transpose(img, (2, 0, 1)), (3, 256, 256), step=self.step)

        sample = {
            "patches": patches[0],
            "ori_img": ori_img,
            "file_name": file_name.split(".")[0] + ".png",
            "mask": mask,
        }
        return sample

    def __len__(self):
        return len(self.imgs)


class AtopyDataset(Dataset):
    def __init__(self, img_path, mask_path, class_path, transforms, idx, skin_path=None):
        self.transforms = transforms
        self.img_path = img_path
        self.mask_path = mask_path
        self.idx = idx
        self.skin_path = skin_path

        # Whole body
        self.patch_size = 645
        self.step = 128

        with open(class_path, "rb") as f:
            data = pickle.load(f)
            index = list(data["index"])
        self.anno = data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], :4]

        self.labellers = [filename for filename in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path, filename))]

    def __getitem__(self, i):
        file = self.idx[i]
        ori_img = cv2.imread(os.path.join(self.img_path, file.split(".")[0] + ".jpg"))
        grade = self.anno[i].astype(np.int64)  # C x labellers(5)

        if self.skin_path is None:
            ori_img = cv2.resize(ori_img, (512, 512))

            # Masks
            masks = []
            for labeller in self.labellers:
                mask = cv2.resize(cv2.imread(os.path.join(self.mask_path, labeller, file.split(".")[0] + ".png"), 0), (512, 512))
                mask[mask != 0] = 1
                masks.append(mask)

            # Augmentation
            transformed = self.transforms(image=ori_img, masks=masks)
            img = transformed["image"]
            masks = np.stack(transformed["masks"], axis=0)  # labellers(2) x H x W

            sample = {"img": img}
        else:  # '20, '21
            # Null background
            skin = cv2.imread(os.path.join(self.skin_path, file.split(".")[0] + ".png"), 0)
            skin = np.stack((skin, skin, skin), axis=-1)  # H x W x 3
            ori_img[skin == 0] = 0

            # Mask
            masks = cv2.imread(os.path.join(self.mask_path, file.split(".")[0] + ".png"), 0)

            # Resize
            dsize = (int(ori_img.shape[1] / self.patch_size * 256), int(ori_img.shape[0] / self.patch_size * 256))
            ori_img = cv2.resize(ori_img, dsize)
            masks = cv2.threshold(cv2.resize(masks, dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :]  # 1 x H x W
            masks[masks != 0] = 1
            skin = cv2.threshold(cv2.resize(skin[..., 0], dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :]  # 1 x H x W
            skin[skin != 0] = 1

            # Pad
            pad_width = (
                (0, self.step - ori_img.shape[0] % self.step),
                (0, self.step - ori_img.shape[1] % self.step),
                (0, 0),
            )
            img = np.pad(ori_img, pad_width)

            # Normalize and patchify
            transform = A.Normalize(mean=(0.4461, 0.5334, 0.7040), std=(0.1161, 0.1147, 0.1204))

            patches = patchify(np.transpose(transform(image=img)["image"], (2, 0, 1)), (3, 256, 256), step=self.step)[
                0
            ]  # Ny x Nx x C x H x W x 3

            sample = {"patches": patches, "mask": skin, "pad_width": np.array(pad_width)}

        sample.update({"seg": masks, "grade": grade, "ori_img": ori_img, "file_name": file.split(".")[0] + ".jpg"})

        return sample

    def __len__(self):
        return len(self.idx)


def get_dataset(dataset_name, split_ratio=0.2):

    # total_num_samples, imgs, masks = get_file_list(phase)

    if dataset_name == "train":
        with open("data/'19/classes/train.txt", "r") as f:
            train_idx, val_ix = train_test_split(f.read().split("\n"), test_size=split_ratio, random_state=42)

        return AtopyDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            train_transforms,
            train_idx,
        ), AtopyDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            val_ix,
        )
    elif dataset_name == "intra":
        with open("data/'19/classes/test.txt", "r") as f:
            test_idx = f.read().split("\n")
        return AtopyDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            test_idx,
        )
    elif dataset_name == "extra":
        return AtopyDataset(
            "data/'19/images/Ext. (SNUBH)",
            "data/'19/labels/Ext. (SNUBH)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'19/labels/Ext. (SNUBH)/동현"),
        )
    elif dataset_name == "2020":
        return AtopyDataset(
            "data/'20 Ext. (SNUBH-adult)/images",
            "data/'20 Ext. (SNUBH-adult)/labels/lesion_area",
            "data/'20 Ext. (SNUBH-adult)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'20 Ext. (SNUBH-adult)/labels/lesion_area"),
            "data/'20 Ext. (SNUBH-adult)/labels/skin_area",
        )
    elif dataset_name == "2021":
        return AtopyDataset(
            "data/'21 Ext. (SNUBH-child)/images",
            "data/'21 Ext. (SNUBH-child)/labels/lesion_area",
            "data/'21 Ext. (SNUBH-child)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'21 Ext. (SNUBH-child)/labels/lesion_area"),
            "data/'21 Ext. (SNUBH-child)/labels/skin_area",
        )
    else:
        raise ValueError()


if __name__ == "__main__":

    cols = 10
    with open("data/'19/classes/train.txt", "r") as f:
        dataset = AtopyDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            train_transforms,
            idx=f.read().split("\n"),
        )

    weights = np.array([np.unique(dataset.anno[:, i], return_counts=True)[1] for i in range(4)])  # C x 4(num_cuts+1)
    weights[:, 0] = weights.sum(1)  # The background class appears in all class
    weights = 1.0 / weights
    weights = weights / weights.sum(1, keepdims=True) * 4  # Normalize
    print("Weights for class-imbalance")
    print(weights)

    train_aug = A.Compose([t for t in train_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])
    test_aug = A.Compose([t for t in test_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])

    mean_h, mean_w = 0, 0
    for i in tqdm(range(len(dataset))):
        img = cv2.imread(os.path.join(dataset.img_path, dataset.idx[i]))
        h, w = img.shape[:2]
        mean_h += h
        mean_w += w
    mean_h /= len(dataset)
    mean_w /= len(dataset)
    print(f"mean size : ({mean_h}, {mean_w})")

    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(64, 16))
    for i in range(cols):
        idx = random.randint(0, len(dataset) - 1)

        img = cv2.imread(os.path.join(dataset.img_path, dataset.idx[i]))
        img = cv2.resize(img, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        masks = []
        for labeller in ["동현", "재성"]:
            mask = cv2.imread(os.path.join("data/'19/labels/Int. (SNU)", labeller, dataset.idx[i].split(".")[0] + ".png"), 0)
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
    ####### COMPUTE MEAN / STD
    test_transforms = A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=0, std=1), ToTensorV2()])
    with open("data/'19/classes/train.txt", "r") as f:
        dataset = AtopyDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            idx=f.read().split("\n"),
        )
    image_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    cohens_k = 0.0

    # loop through images
    for data in tqdm(image_loader):
        inputs = data["img"]
        mask = data["seg"]
        grade = data["grade"]
        psum += inputs.sum(axis=[0, 2, 3])
        psum_sq += (inputs**2).sum(axis=[0, 2, 3])
        mask = mask.numpy()
        for i in range(mask.shape[0]):
            k = cohen_kappa_score(mask[i, 0].reshape(-1), mask[i, 1].reshape(-1))
            if not np.isnan(k):
                cohens_k += k

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
