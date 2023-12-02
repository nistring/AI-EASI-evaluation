from typing import List
from typing import Dict
import itertools

import torch
import cv2
import numpy as np


def draw_results(image: np.ndarray, mask: torch.Tensor, categories: List[str]):
    assert mask.shape[0] == len(categories)

    colours = (
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 128, 255),
        (0, 255, 128),
        (128, 0, 255),
    )

    for label, (category, category_mask) in enumerate(zip(categories, mask)):

        cat_image = image.copy()
        mask_image = image.copy()
        mask_image[~category_mask] = 0

        cat_colour = colours[label % len(colours)]
        cat_colour = np.array(cat_colour)
        cat_image[category_mask] = 0.5 * cat_image[category_mask] + 0.5 * cat_colour

        yield category, cat_image, mask_image
