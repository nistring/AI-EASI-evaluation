import argparse
import logging
import pathlib
import functools

import cv2
import torch
from torchvision import transforms
import os
from scipy.ndimage import zoom
import numpy as np

from semantic_segmentation import models
from semantic_segmentation import load_model
from semantic_segmentation import draw_results

fn_image_transform = transforms.Compose(
    [
        transforms.Lambda(
            lambda x: cv2.resize(x[0], (int(x[0].shape[1] * x[1] // 32) * 32, int(x[0].shape[0] * x[1] // 32) * 32))
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--model-type", type=str, choices=models, required=True)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--scale", type=float, default=256 / 578)

    return parser.parse_args()


def find_files(dir_path: pathlib.Path, file_exts):
    assert dir_path.exists()
    assert dir_path.is_dir()

    for file_ext in file_exts:
        yield from dir_path.rglob(f"*{file_ext}")


def _load_image(image_path: pathlib.Path, scale: float):
    image = cv2.imread(str(image_path))
    return cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"running inference on {device}")

    assert args.display or args.save_dir

    logging.info(f"loading {args.model_type} from {args.model}")
    model = torch.load(args.model, map_location=device)
    model = load_model(models[args.model_type], model)
    model.to(device).eval()

    logging.info(f"evaluating images from {args.images}")
    image_dir = pathlib.Path(args.images)


    for image_file in find_files(image_dir, [".png", ".jpg", ".jpeg", ".JPG"]):
        logging.info(f"segmenting {image_file} with threshold of {args.threshold}")

        ori_img = _load_image(image_file, args.scale)
        image = fn_image_transform((cv2.cvtColor(ori_img, cv2.COLOR_RGB2BRG), 640 * 2 / sum(ori_img.shape)))

        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            results = model(image)["out"][0]
            results = torch.sigmoid(results).cpu().numpy()
            results = (results > args.threshold).astype("uint8") + 2

        image = cv2.resize(ori_img, (image.shape[3], image.shape[2]))
        for i in range(results.shape[0]):
            mask = results[i]
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
            results[i] = np.where((mask == 2) | (mask == 0), 0, 1)

        results = zoom(results.astype("float"), [1] + [a/b for a, b in zip(ori_img.shape[:2], results.shape[1:])])
        results = results > 0.5
        for category, category_image, mask_image in draw_results(ori_img, results, categories=model.categories):
            if args.save_dir:
                output_name = f"results_{category}_{image_file.name}"
                logging.info(f"writing output to {output_name}")
                cv2.imwrite(os.path.join(args.save_dir, str(output_name)), category_image)
                cv2.imwrite(os.path.join(args.save_dir, f"mask_{category}_{image_file.name}"), mask_image)

            if args.display:
                cv2.imshow(category, category_image)
                cv2.imshow(f"mask_{category}", mask_image)

        if args.display:
            if cv2.waitKey(0) == ord("q"):
                logging.info("exiting...")
                exit()
