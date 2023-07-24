from argparse import ArgumentParser
import os
import cv2
import numpy as np
from copy import deepcopy
from utils import *
from tqdm import tqdm

if __name__ == "__main__":
    # arguments
    parser = ArgumentParser()
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--std-th", type=float, default=0.0, help="Threshold for std metric in z score")
    parser.add_argument("--entropy-th", type=float, default=0.0, help="Threshold for entropy metric in z score")
    parser.add_argument("--mi-th", type=float, default=0.0, help="Threshold for mutual information metric in z score")
    args = parser.parse_args()

    # make vis directory
    root = os.path.join("results", args.exp_name)
    os.makedirs(os.path.join(root, "vis"), exist_ok=True)

    # load results
    with open(os.path.join(root, "results.pickle"), "rb") as f:
        results = pickle.load(f)

    print(f"Visualize {args.exp_name}")
    for i in tqdm(range(results["img"].shape[0])):
        # Load
        img = results["img"][i]
        gt = results["gt"][i]
        gt_grade = results["gt_grade"][i]
        mean = results["mean"][i]
        weighted_mean = results["weighted_mean"][i]
        std = results["std"][i]
        entropy = results["entropy"][i]
        mi = results["mi"][i]
        grade = results["grade"][i]
        cohens_k = results["cohens_k"][i]

        # Size
        height, width = img.shape[:2]
        height = height // 2
        width = width // 2
        scale = 0.5

        # Normalize uncertainty
        std = 1 - normalize_uncertainty(std)
        entropy = 1 - normalize_uncertainty(entropy)
        mi = 1 - normalize_uncertainty(mi)

        # Resize
        img = cv2.resize(img, (width, height))
        mean = cv2.resize(mean, (width, height))
        weighted_mean = cv2.resize(weighted_mean, (width, height))
        std = cv2.resize(std, (width, height))
        entropy = cv2.resize(entropy, (width, height))
        mi = cv2.resize(mi, (width, height))

        # Thresholding
        mean_mask = mean >= 0.5
        std_mask = std > std.mean() - args.std_th * std.std()
        entropy_mask = entropy > entropy.mean() - args.entropy_th * entropy.std()
        mi_mask = mi > mi.mean() - args.mi_th * mi.std()

        # Draw contours
        std_pred, std_area, std_easi = draw_contours(deepcopy(img), mean_mask, std_mask, weighted_mean)
        entropy_pred, entropy_area, entropy_easi = draw_contours(deepcopy(img), mean_mask, entropy_mask, weighted_mean)
        mi_pred, mi_area, mi_easi = draw_contours(deepcopy(img), mean_mask, mi_mask, weighted_mean)

        # Gray to RGB
        mean = (cv2.cvtColor(mean, cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)
        std = (cv2.cvtColor(std, cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)
        entropy = (cv2.cvtColor(entropy, cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)
        mi = (cv2.cvtColor(mi, cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)
        gt = (cv2.cvtColor(np.clip((gt / 2).astype(np.float32), 0, 1), cv2.COLOR_GRAY2RGB) * 255).astype(np.uint8)

        # Information
        info = np.zeros_like(mean)
        x, y = int(10 * scale), int(25 * scale)
        dy = int(25 * scale)
        cv2.putText(
            info,
            f"Ground truth grade : {gt_grade}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"Predicted grade : {grade.mean():.2f}(std={grade.std():.2f})",
            (x, y + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"std(sigma={args.std_th:.2f}); area : {std_area[0]:.1f}-{std_area[1]:.1f}, EASI : {std_easi[0]:.1f}-{std_easi[1]:.1f}",
            (x, y + 2 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"entropy(sigma={args.entropy_th:.2f}); area : {entropy_area[0]:.1f}-{entropy_area[1]:.1f}, EASI : {entropy_easi[0]:.1f}-{entropy_easi[1]:.1f}",
            (x, y + 3 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"mi(sigma={args.mi_th:.2f}); area : {mi_area[0]:.1f}-{mi_area[1]:.1f}, EASI : {mi_easi[0]:.1f}-{mi_easi[1]:.1f}",
            (x, y + 4 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )
        cv2.putText(
            info,
            f"Cohen's K : {cohens_k:.2f}",
            (x, y + 5 * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale / 2,
            (255, 255, 255),
            int(scale),
            cv2.LINE_AA,
        )

        # Merge images
        img = np.concatenate(
            (np.concatenate((gt, mean, std, entropy, mi), axis=1), np.concatenate((img, info, std_pred, entropy_pred, mi_pred), axis=1)),
            axis=0,
        )

        # Write
        cv2.imwrite(os.path.join(root, "vis", str(i).zfill(4) + ".png"), img)
