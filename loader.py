import os
import cv2
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from patchify import patchify
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

from transforms import *
from utils import area2score

PATCH_SIZE_MULTIPLIER = 1.5
BASE_PATCH_SIZE = 256

class WholeBodyPredictDataset(Dataset):
    def __init__(self, img_dir_path, mask_path, transforms=test_transforms, step_size=96):
        """Whole-body image dataset for prediction step

        Args:
            img_dir_path (str): Directory path where images are stored.
            mask_path (str): Directory path where masks are stored.
            transforms (callable, optional): A function/transform to apply to the images. Defaults to test_transforms.
            step_size (int, optional): Defaults to 96.
        """
        self.transforms = transforms
        self.img_dir_path = img_dir_path
        self.mask_path = mask_path
        self.img_path = os.listdir(img_dir_path)

        # Whole body
        self.patch_size = BASE_PATCH_SIZE * PATCH_SIZE_MULTIPLIER  # 645 * 1.5
        self.step = step_size

    def __getitem__(self, i):
        # Load
        ori_img = cv2.imread(str(Path(self.img_dir_path) / self.img_path[i]))
        mask = cv2.imread(str(Path(self.mask_path) / self.img_path[i]), 0)

        # Masking
        ori_img = ori_img * (mask > 0)[:, :, np.newaxis]
        x_flat = mask.any(0).nonzero()[0]
        x_min, x_max = x_flat[0], x_flat[-1]
        y_flat = mask.any(1).nonzero()[0]
        y_min, y_max = y_flat[0], y_flat[-1]

        # Crop
        ori_img = ori_img[y_min : y_max + 1, x_min : x_max + 1]
        mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        # Resize
        dsize = (int(ori_img.shape[1] / self.patch_size * 256), int(ori_img.shape[0] / self.patch_size * 256))
        ori_img = cv2.resize(ori_img, dsize)
        mask = cv2.threshold(cv2.resize(mask, dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :] > 0  # 1HW

        # Pad
        pad_width = (
            (self.step, 2 * self.step - (ori_img.shape[0] - 256) % self.step if ori_img.shape[0] > 256 else 256 - ori_img.shape[0]),
            (self.step, 2 * self.step - (ori_img.shape[1] - 256) % self.step if ori_img.shape[1] > 256 else 256 - ori_img.shape[1]),
            (0, 0),
        )
        img = cv2.cvtColor(np.pad(ori_img, pad_width), cv2.COLOR_BGR2RGB)

        # Normalize and patchify
        patches = norm_transforms(
            self.transforms(torch.Tensor(patchify(np.transpose(img, (2, 0, 1)), (3, 256, 256), step=self.step)[0])) / 255
        )  # NyNxC3HW

        return {
            "patches": patches,
            "mask": mask,
            "pad_width": np.array(pad_width),
            "ori_img": ori_img,
            "file_name": self.img_path[i].split(".")[0] + ".jpg",
        }

    def __len__(self):
        return len(self.img_path)


class ROIPredict(Dataset):
    def __init__(self, img_dir_path, transforms=test_transforms):
        """Whole-body image dataset for prediction step

        Args:
            img_dir_path (str): Directory path where images are stored.
            transforms (optional): Defaults to test_transforms.
        """
        self.img_path = os.listdir(img_dir_path)
        self.img_dir_path = img_dir_path
        self.transforms = transforms

    def __getitem__(self, i):
        img = read_image(str(Path(self.img_dir_path) / self.img_path[i]))  # 3HW
        img = self.transforms(img)
        ori_img = torch.permute(torch.flip(img, dims=(0,)), dims=(1, 2, 0))  # 3(RGB)HW -> HW3(BGR)
        img = norm_transforms(img)

        return {"img": img, "ori_img": ori_img, "file_name": self.img_path[i].split(".")[0] + ".jpg"}

    def __len__(self):
        return len(self.img_path)

class WholeBodyTestDataset(Dataset):
    def __init__(self, img_path, seg_path, class_path, transforms, idx, mask_path, step_size):
        """Dataset for inference step in whole-body model

        Args:
            img_path (str): Directory path where images are stored.
            seg_path (str): Directory path where segment labelse are stored.
            class_path (str): Path where the severity class information is stored.
            transforms (callable): A function/transform to apply to the images.
            idx (List): List of individual image file paths.
            mask_path (str): Directory path where masks are stored.
            step_size (int):
        """
        self.transforms = transforms
        self.img_path = img_path
        self.seg_path = seg_path
        self.idx = idx
        self.mask_path = mask_path

        # Whole body
        self.patch_size = 256 * 1.5  # 645 * 1.5
        self.step = step_size

        with open(class_path, "rb") as f:
            data = pickle.load(f)
            index = list(data["index"])
        self.anno = data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], :4]
        self.area = (
            data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], 4] if self.seg_path is None else None
        )

    def __getitem__(self, i):
        # Load
        file = self.idx[i]
        ori_img = cv2.imread(str(Path(self.img_path) / (file.split(".")[0] + ".png")))  # ".jpg")))
        grade = torch.LongTensor(self.anno[i]).T  # lC
        mask = cv2.imread(str(Path(self.mask_path) / (file.split(".")[0] + ".png")), 0)

        # Masking
        ori_img = ori_img * (mask > 0)[:, :, np.newaxis]
        x_flat = mask.any(0).nonzero()[0]
        x_min, x_max = x_flat[0], x_flat[-1]
        y_flat = mask.any(1).nonzero()[0]
        y_min, y_max = y_flat[0], y_flat[-1]

        # Crop
        ori_img = ori_img[y_min : y_max + 1, x_min : x_max + 1]
        mask = mask[y_min : y_max + 1, x_min : x_max + 1]

        # Resize
        dsize = (int(ori_img.shape[1] / self.patch_size * 256), int(ori_img.shape[0] / self.patch_size * 256))
        ori_img = cv2.resize(ori_img, dsize)
        mask = cv2.threshold(cv2.resize(mask, dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :] > 0  # 1HW

        # Area score
        if self.seg_path is None:
            area = self.area[[i]]
            seg = np.zeros(dsize[::-1], dtype=bool)[np.newaxis, :, :]
        else:
            seg = cv2.imread(str(Path(self.seg_path) / (file.split(".")[0] + ".png")), 0)
            seg = seg[y_min : y_max + 1, x_min : x_max + 1]
            seg = cv2.threshold(cv2.resize(seg, dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :] > 0  # 1HW
            area = area2score(seg.astype(np.float32).mean((1, 2)) / mask.astype(np.float32).mean((1, 2)))

        # Pad
        pad_width = (
            (self.step, 2 * self.step - (ori_img.shape[0] - 256) % self.step if ori_img.shape[0] > 256 else 256 - ori_img.shape[0]),
            (self.step, 2 * self.step - (ori_img.shape[1] - 256) % self.step if ori_img.shape[1] > 256 else 256 - ori_img.shape[1]),
            (0, 0),
        )
        img = cv2.cvtColor(np.pad(ori_img, pad_width), cv2.COLOR_BGR2RGB)

        # Normalize and patchify
        patches = norm_transforms(
            self.transforms(torch.Tensor(patchify(np.transpose(img, (2, 0, 1)), (3, 256, 256), step=self.step)[0])) / 255
        )  # NyNxC3HW
        return {
            "patches": patches,
            "mask": mask,
            "pad_width": np.array(pad_width),
            "seg": seg,
            "area": area,
            "grade": grade,
            "ori_img": ori_img,
            "file_name": file.split(".")[0] + ".jpg",
        }

    def __len__(self):
        return len(self.idx)


class WholeBodyTrainDataset(Dataset):
    def __init__(self, img_path, transforms, idx):
        """Dataset for training whole-body model.

        Args:
            img_path (str): Directory path where images are stored.
            transforms (callable): A function/transform that takes in an image and returns a transformed version.
            idx (List): List of individual image file paths.
        """
        self.transforms = transforms
        self.img_path = img_path
        self.idx = idx

    def __getitem__(self, i):
        img = norm_transforms(self.transforms(read_image(str(Path(self.img_path) / self.idx[i]))))  # N3HW
        seg = torch.zeros((img.shape[0], 4, img.shape[2], img.shape[3]), dtype=torch.int64)
        return {"img": img, "seg": seg}

    def __len__(self):
        return len(self.idx)


class ROIDataset(Dataset):
    def __init__(self, img_path, seg_path, class_path, transforms, idx, use_synthetic=False):
        """Dataset for training ROI model and whole-body model.

        Args:
            img_path (str): Directory path where images are stored.
            seg_path (str): Directory path where segment labelse are stored.
            class_path (str): Path where the severity class information is stored.
            transforms :
            idx (List): List of individual image file paths.
        """
        self.transforms = transforms
        self.img_path = img_path
        self.seg_path = seg_path
        self.idx = idx
        self.use_synthetic = use_synthetic

        with open(class_path, "rb") as f:
            data = pickle.load(f)
            index = list(data["index"])
        self.anno = data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], :4]

        self.labellers = ["동현", "재성"]

    def __getitem__(self, i):
        file = self.idx[i]
        img = read_image(str(Path(self.img_path) / (file.split(".")[0] + ".jpg")))  # 3HW
        grade = torch.LongTensor(self.anno[i]).T  # L'C
        seg = torch.stack(
            [read_image(str(Path(self.seg_path) / l / (file.split(".")[0] + ".png")), ImageReadMode.GRAY) for l in self.labellers]
        )  # L1HW

        seg = tv_tensors.Mask(seg)
        img, seg = self.transforms(img, seg)
        ori_img = torch.permute(torch.flip(img, dims=(0,)), dims=(1, 2, 0))  # 3(RGB)HW -> HW3(BGR)
        img = norm_transforms(img)
        seg = seg.bool().squeeze(1)  # LHW

        data = {
            "img": img,
            "seg": seg,
            "area": area2score(seg.float().mean((1, 2))),
            "grade": grade,
            "ori_img": ori_img,
            "file_name": file.split(".")[0] + ".jpg",
        }
        if self.use_synthetic:
            data["syn"] = self.transforms(tv_tensors.Mask(np.load(str(Path("results") / "synthetic" / (file.split(".")[0] + ".npy")))))

        return data

    def __len__(self):
        return len(self.idx)


def get_dataset(dataset_name, split_ratio=0.2, step_size=128, wholebody=False, synthetic=False):
    """Return a specific dataset requested.

    Args:
        dataset_name (str): The name of the dataset.
        split_ratio (float, optional): Split ratio of the training to tuning(val) dataset. Defaults to 0.2.
        step_size (int, optional): Defaults to 128.
        wholebody (bool, optional): Use a different image transformation to ROI images when training whole-body model. Defaults to False.

    Raises:
        ValueError: Raise error if the requested dataset is not listed.

    Returns:
        Dataset
    """

    # Train datasets
    if dataset_name == "roi_train":
        with open("data/'19/classes/train.txt", "r") as f:
            train_idx, val_idx = train_test_split(f.read().split("\n"), test_size=split_ratio, random_state=42)

        return ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            roi_train_transforms if wholebody else train_transforms,
            train_idx,
            synthetic,
        ), ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            roi_test_transforms if wholebody else test_transforms,
            val_idx,
            synthetic,
        )
    elif dataset_name == "wb_train":
        train_idx, val_idx = train_test_split(os.listdir("data/'20 Int. (SNU-adult)/images/NL"), test_size=split_ratio, random_state=42)
        return WholeBodyTrainDataset(
            "data/'20 Int. (SNU-adult)/images/NL",
            wholebody_train_transforms,
            train_idx,
        ), WholeBodyTrainDataset(
            "data/'20 Int. (SNU-adult)/images/NL",
            wholebody_test_transforms,
            val_idx,
        )

    # Test datasets
    elif dataset_name == "19_int":
        if synthetic:
            with open("data/'19/classes/train.txt", "r") as f:
                test_idx = f.read().split("\n")
        else:
            with open("data/'19/classes/test.txt", "r") as f:
                test_idx = f.read().split("\n")
        return ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            test_idx,
        )
    elif dataset_name == "19_ext":
        return ROIDataset(
            "data/'19/images/Ext. (SNUBH)",
            "data/'19/labels/Ext. (SNUBH)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'19/labels/Ext. (SNUBH)/동현"),
        )
    elif dataset_name == "20_int":
        return WholeBodyTestDataset(
            "data/'20 Int. (SNU-adult)/images/AD",
            None,
            "data/'20 Int. (SNU-adult)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'20 Int. (SNU-adult)/images/AD"),
            "data/'20 Int. (SNU-adult)/masks/AD",
            step_size,
        )
    elif dataset_name == "20_ext":
        return WholeBodyTestDataset(
            "data/'20 Ext. (SNUBH-adult)/selected",
            None,  # "data/'20 Ext. (SNUBH-adult)/labels/lesion_area",
            "data/'20 Ext. (SNUBH-adult)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'20 Ext. (SNUBH-adult)/selected"),
            "data/'20 Ext. (SNUBH-adult)/masks",
            step_size,
        )

    # Prediction datasets
    elif dataset_name == "roi_predict":
        return ROIPredict("data/roi_predict")
    elif dataset_name == "wb_predict":
        return WholeBodyPredictDataset("data/wb_predict/image", "data/wb_predict/mask")
    else:
        raise ValueError()


if __name__ == "__main__":

    with open("data/'19/classes/train.txt", "r") as f:
        dataset = ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            train_transforms,
            idx=f.read().split("\n"),
        )

    weights = np.array([np.unique(dataset.anno[:, i], return_counts=True)[1] for i in range(4)])  # Cc(num_cuts+1)
    weights[:, 0] = weights[:, 0]  #  + weights.sum(1)  # whole body (divided by the number of seg labels)
    weights = 1.0 / weights
    weights = weights / weights.sum(1, keepdims=True) * 4  # Normalize
    weights = 0.5 * (1.0 + weights)
    print("Weights for class-imbalance")
    print(weights)

    # mean_h, mean_w = 0, 0
    # for i in tqdm(range(len(dataset))):
    #     img = cv2.imread(os.path.join(dataset.img_path, dataset.idx[i]))
    #     h, w = img.shape[:2]
    #     mean_h += h
    #     mean_w += w
    # mean_h /= len(dataset)
    # mean_w /= len(dataset)
    # print(f"mean size : ({mean_h}, {mean_w})")

    """
    Adopted from https://kozodoi.me/blog/20210308/compute-image-stats
    """
    ####### COMPUTE MEAN / STD
    image_size = 256
    test_transforms = v2.Compose(
        [
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32),
            Scale(),
            v2.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ]
    )
    with open("data/'19/classes/train.txt", "r") as f:
        dataset = ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            idx=f.read().split("\n"),
        )
    image_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=47, pin_memory=True)
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for data in tqdm(image_loader):
        inputs = data["img"]
        mask = data["seg"]
        grade = data["grade"]
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
