import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from scipy.ndimage import zoom
import sys
import time
from utils import *


sys.path.append("SemanticSegmentation")
from evaluate_images import fn_image_transform
from streamlit_image_coordinates import streamlit_image_coordinates


def grab_cut(image, mask):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1)
    return mask


st.title("AI supported EASI estimation")

with st.form(key="input"):
    uploaded_file = st.file_uploader("Upload an image.")
    scale = st.slider("Scale", 0.1, 1.0, 256 / 578)
    threshold = st.slider("Threshold", 0.0, 1.0, 0.2)
    if uploaded_file is not None:
        ori_img = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    input_submit = st.form_submit_button()


if input_submit:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f"Running inference on {device}"
    ori_img = cv2.cvtColor(cv2.resize(ori_img, (int(ori_img.shape[1] * scale), int(ori_img.shape[0] * scale))), cv2.COLOR_BGR2RGB)

    model = torch.jit.load("weights/BiSeNetV2.pt").to(device).half().eval()

    image = fn_image_transform((ori_img, 640 * 2 / sum(ori_img.shape)))

    with torch.no_grad():
        image = image.to(device).unsqueeze(0).half()
        results = model(image)[0, 0]
        results = torch.sigmoid(results).cpu().numpy()
        results = (results > threshold).astype("uint8") + 2

    image = cv2.resize(ori_img, (image.shape[3], image.shape[2]))
    results = grab_cut(image, results).astype("bool")

    grab_img = st.empty()
    cat_image = image.copy()
    cat_colour = np.array([255, 0, 0])
    cat_image[results] = 0.5 * cat_image[results] + 0.5 * cat_colour
    grab_img.image(cat_image)

    if st.button("Predict"):

        results = zoom(results.astype("float"), [1] + [a/b for a, b in zip(ori_img.shape[:2], results.shape[1:])])
        results = results > 0.5
        xmin, xmax = np.non_zero(results.sum(0))[[0, -1]]
        ymin, ymax = np.non_zero(results.sum(1))[[0, -1]]
        width = ((xmax - xmin) // 256 + 1) * 256
        height = ((xmax - xmin) // 256 + 1) * 256
        # cat_image = ori_img.copy()
        # cat_colour = np.array([255, 0, 0])
        # cat_image[results] = 0.5 * cat_image[results] + 0.5 * cat_colour

        # st.image(cat_image)

        # results = zoom(results.astype("float"), [a/b for a, b in zip(ori_img.shape[:2], results.shape)])
        # results = results > 0.5


# model = MyModel().eval().cuda()
# inputs = [torch.randn((1, 3, 224, 224)).cuda()]
# trt_ts = torch_tensorrt.compile(model, ir="ts", inputs) # Output is a ScriptModule object
# torch.jit.save(trt_ts, "trt_model.ts")
