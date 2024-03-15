from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from patchify import patchify
from sklearn.metrics import cohen_kappa_score
import pickle
from sklearn.model_selection import train_test_split
from transforms import wholebody_transforms, test_transforms, train_transforms
from torchvision.io import read_image, ImageReadMode
from pathlib import Path

image_size = 256

class WholeBodyTestDataset(Dataset):
    def __init__(self, img_path, seg_path, class_path, transforms, idx, mask_path):
        self.transforms = transforms
        self.img_path = img_path
        self.seg_path = seg_path
        self.idx = idx
        self.mask_path = mask_path

        # Whole body
        self.patch_size = 645
        self.step = 128

        with open(class_path, "rb") as f:
            data = pickle.load(f)
            index = list(data["index"])
        self.anno = data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], :4]

    def __getitem__(self, i):
        # Load
        file = self.idx[i]
        ori_img = cv2.imread(str(Path(self.img_path) / file.split(".")[0] + ".jpg"))
        seg = cv2.imread(str(Path(self.seg_path) / file.split(".")[0] + ".png"), 0)
        grade = torch.LongTensor(self.anno[i])  # Cl
        mask = cv2.imread(str(Path(self.mask_path) / file.split(".")[0] + ".png"), 0)

        # Masking
        ori_img = ori_img * (mask > 0)[:,:,np.newaxis]
        x_min, x_max = mask[0].any(1).nonzero()[[0, -1]]
        y_min, y_max = mask[1].any(1).nonzero()[[0, -1]]

        # Crop
        ori_img = ori_img[y_min : y_max + 1, x_min : x_max + 1]
        seg = seg[y_min : y_max + 1, x_min : x_max + 1]

        # Resize
        dsize = (int(ori_img.shape[1] / self.patch_size * 256), int(ori_img.shape[0] / self.patch_size * 256))
        ori_img = cv2.resize(ori_img, dsize)
        seg = cv2.threshold(cv2.resize(seg, dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :]  # 1 x H x W
        seg = (seg > 0).astype(float)
        mask = cv2.threshold(cv2.resize(mask[..., 0], dsize), 127, 255, cv2.THRESH_BINARY)[1][np.newaxis, :, :]  # 1 x H x W
        mask = (mask > 0).astype(float)

        # Pad
        pad_width = (
            (0, self.step - ori_img.shape[0] % self.step),
            (0, self.step - ori_img.shape[1] % self.step),
            (0, 0),
        )
        img = np.pad(ori_img, pad_width)

        # Normalize and patchify
        # patches = patchify(np.transpose(transform(image=img)["image"], (2, 0, 1)), (3, 256, 256), step=self.step)[
        #     0
        # ]  # Ny x Nx x C x H x W x 3
        patches = self.transforms(patchify(np.transpose(img, (2, 0, 1)), (3, 256, 256), step=self.step) / 255) # NyNxC3HW

        return {"patches": patches, "mask": mask, "pad_width": np.array(pad_width), "seg": seg, "grade": grade, "ori_img": ori_img, "file_name": file.split(".")[0] + ".jpg"}

    def __len__(self):
        return len(self.idx)


class WholeBodyTrainDataset(Dataset):
    def __init__(self, img_path, transforms, idx):
        self.transforms = transforms
        self.img_path = img_path
        self.idx = idx

    def __getitem__(self, i):
        file = self.idx[i]
        img = read_image(str(Path(self.img_path) /  file))

        img = self.transforms(img/255) # CHW
        seg = torch.zeros_like(img, dtype=bool)[:1] # 1HW
        grade = torch.zeros((4, 1), dtype=torch.int64) # Cl

        return {"img" : img, "seg" : seg, "grade" : grade}

    def __len__(self):
        return len(self.idx)


class ROIDataset(Dataset):
    def __init__(self, img_path, seg_path, class_path, transforms, idx):
        self.transforms = transforms
        self.img_path = img_path
        self.seg_path = seg_path
        self.idx = idx

        with open(class_path, "rb") as f:
            data = pickle.load(f)
            index = list(data["index"])
        self.anno = data["annotations"][[index.index(os.path.basename(x).split(".")[0]) for x in self.idx], :4]

        self.labellers = [filename for filename in os.listdir(seg_path) if os.path.isdir(os.path.join(seg_path, filename))]

    def __getitem__(self, i):
        file = self.idx[i]
        img = read_image(str(Path(self.img_path) / (file.split(".")[0] + ".jpg")))
        grade = torch.LongTensor(self.anno[i])  # Cl
        seg = torch.cat([read_image(str(Path(self.seg_path) / l / (file.split(".")[0] + ".png")), ImageReadMode.GRAY) for l in self.labellers])

        try:
            from torchvision import tv_tensors
            seg = tv_tensors.Mask(seg)
        except:
            from torchvision import datapoints
            seg = datapoints.Mask(seg)

        img, seg = self.transforms(img/255, seg)

        return {"img" : img, "seg" : seg.bool(), "grade" : grade}

    def __len__(self):
        return len(self.idx)


def get_dataset(dataset_name, split_ratio=0.2):

    # total_num_samples, imgs, masks = get_file_list(phase)

    if dataset_name == "roi_train":
        with open("data/'19/classes/train.txt", "r") as f:
            train_idx, val_idx = train_test_split(f.read().split("\n"), test_size=split_ratio, random_state=42)

        return ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            train_transforms,
            train_idx,
        ), ROIDataset(
            "data/'19/images/Int. (SNU)",
            "data/'19/labels/Int. (SNU)",
            "data/'19/classes/meta_result.pkl",
            test_transforms,
            val_idx,
        )
    elif dataset_name == "wb_train":
        train_idx, val_idx = train_test_split(os.listdir("data/'20 Int. (SNU-adult)/images/NL"), test_size=split_ratio, random_state=42)
        return WholeBodyTrainDataset(
            "data/'20 Int. (SNU-adult)/images/NL",
            wholebody_transforms,
            train_idx,
        ), WholeBodyTrainDataset(
            "data/'20 Int. (SNU-adult)/images/NL",
            wholebody_transforms,
            val_idx,
        )
    elif dataset_name == "19_int":
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
    elif dataset_name == "20_ext":
        return WholeBodyTestDataset(
            "data/'20 Ext. (SNUBH-adult)/images",
            "data/'20 Ext. (SNUBH-adult)/labels/lesion_area",
            "data/'20 Ext. (SNUBH-adult)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'20 Ext. (SNUBH-adult)/labels/lesion_area"),
            "data/'20 Ext. (SNUBH-adult)/labels/skin_area",
        )
    elif dataset_name == "21_ext":
        return WholeBodyTestDataset(
            "data/'21 Ext. (SNUBH-child)/images",
            "data/'21 Ext. (SNUBH-child)/labels/lesion_area",
            "data/'21 Ext. (SNUBH-child)/classes/meta_result.pkl",
            test_transforms,
            os.listdir("data/'21 Ext. (SNUBH-child)/labels/lesion_area"),
            "data/'21 Ext. (SNUBH-child)/labels/skin_area",
        )
    else:
        raise ValueError()


# if __name__ == "__main__":

#     cols = 10
#     with open("data/'19/classes/train.txt", "r") as f:
#         dataset = ROIDataset(
#             "data/'19/images/Int. (SNU)",
#             "data/'19/labels/Int. (SNU)",
#             "data/'19/classes/meta_result.pkl",
#             train_transforms,
#             idx=f.read().split("\n"),
#         )

#     weights = np.array([np.unique(dataset.anno[:, i], return_counts=True)[1] for i in range(4)])  # C x 4(num_cuts+1)
#     weights[:, 0] = weights.sum(1)  # The background class appears in all class
#     weights = 1.0 / weights
#     weights = weights / weights.sum(1, keepdims=True) * 4  # Normalize
#     print("Weights for class-imbalance")
#     print(weights)

#     train_aug = v2.Compose([t for t in train_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])
#     test_aug = v2.Compose([t for t in test_transforms if not isinstance(t, (A.Normalize, ToTensorV2))])

#     mean_h, mean_w = 0, 0
#     for i in tqdm(range(len(dataset))):
#         img = cv2.imread(os.path.join(dataset.img_path, dataset.idx[i]))
#         h, w = img.shape[:2]
#         mean_h += h
#         mean_w += w
#     mean_h /= len(dataset)
#     mean_w /= len(dataset)
#     print(f"mean size : ({mean_h}, {mean_w})")

#     fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(64, 16))
#     for i in range(cols):
#         idx = random.randint(0, len(dataset) - 1)

#         img = cv2.imread(os.path.join(dataset.img_path, dataset.idx[i]))
#         img = cv2.resize(img, (512, 512))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         masks = []
#         for labeller in ["동현", "재성"]:
#             mask = cv2.imread(os.path.join("data/'19/labels/Int. (SNU)", labeller, dataset.idx[i].split(".")[0] + ".png"), 0)
#             mask = cv2.resize(mask, (512, 512))
#             mask[mask != 0] = 1
#             masks.append(mask)

#         before = test_aug(image=img, masks=masks)
#         mask1, mask2 = before["masks"]
#         mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
#         mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
#         before = np.concatenate([before["image"], mask1, mask2], axis=0)

#         after = train_aug(image=img, masks=masks)
#         mask1, mask2 = after["masks"]
#         mask1 = np.stack([mask1, mask1, mask1], axis=-1) * 255
#         mask2 = np.stack([mask2, mask2, mask2], axis=-1) * 255
#         after = np.concatenate([after["image"], mask1, mask2], axis=0)

#         img = np.concatenate([before, after], axis=1)

#         ax[i].imshow(img)
#         ax[i].set_axis_off()
#         ax[i].set_title(f"Image {i}", fontsize=32)

#     plt.tight_layout()
#     fig.savefig("augmentation.png")

#     """
#     Adopted from https://kozodoi.me/blog/20210308/compute-image-stats
#     """
#     ####### COMPUTE MEAN / STD
#     test_transforms = A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=0, std=1), ToTensorV2()])
#     with open("data/'19/classes/train.txt", "r") as f:
#         dataset = AtopyDataset(
#             "data/'19/images/Int. (SNU)",
#             "data/'19/labels/Int. (SNU)",
#             "data/'19/classes/meta_result.pkl",
#             test_transforms,
#             idx=f.read().split("\n"),
#         )
#     image_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=16, pin_memory=True)
#     # placeholders
#     psum = torch.tensor([0.0, 0.0, 0.0])
#     psum_sq = torch.tensor([0.0, 0.0, 0.0])
#     cohens_k = 0.0

#     # loop through images
#     for data in tqdm(image_loader):
#         inputs = data["img"]
#         mask = data["seg"]
#         grade = data["grade"]
#         psum += inputs.sum(axis=[0, 2, 3])
#         psum_sq += (inputs**2).sum(axis=[0, 2, 3])
#         mask = mask.numpy()
#         for i in range(mask.shape[0]):
#             k = cohen_kappa_score(mask[i, 0].reshape(-1), mask[i, 1].reshape(-1))
#             if not np.isnan(k):
#                 cohens_k += k

#     ####### FINAL CALCULATIONS

#     # pixel count
#     count = len(dataset) * image_size * image_size

#     # mean and std
#     total_mean = psum / count
#     total_var = (psum_sq / count) - (total_mean**2)
#     total_std = torch.sqrt(total_var)
#     total_cohen = cohens_k / len(dataset)

#     # output
#     print("mean: " + str(total_mean))
#     print("std:  " + str(total_std))
#     print("cohen's k:  " + str(total_cohen))
