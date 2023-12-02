import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from scipy.ndimage import zoom
import sys
from utils import *
from patchify import patchify
import albumentations as A
from model.model_utils import log_cumulative
import torch.nn.functional as F
import gc

sys.path.append("SemanticSegmentation")
from evaluate_images import fn_image_transform
from streamlit_image_coordinates import streamlit_image_coordinates

torch.cuda.empty_cache()
step = 192
num_classes = 4
device = "cuda" if torch.cuda.is_available() else "cpu"
cutpoints = torch.HalfTensor([[2.4492, 5.1406], [1.3184, 3.3164], [1.4512, 3.1914], [1.4121, 3.0527]]).to(device)
st.set_page_config(layout="wide")
f"Running inference on {device}"
if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.on_change = False
    st.session_state.coord = None
    st.session_state.inference = False
    st.session_state.uploaded_file = None


def on_change_cb():
    st.session_state.on_change = True


def click_button():
    st.session_state.inference = True


def grab_cut(image, mask):
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask == 2) | (mask == 0), 0, 1)
    return mask


# Title
st.title("AI supported EASI estimation")

# Upload
uploaded_file = st.file_uploader("Upload an image.")
scale = st.sidebar.slider("Scale", 0.01, 0.5, 256 / 578 / 4)
threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.2)

if uploaded_file is not None:
    # BiSeNetV2
    if uploaded_file != st.session_state.uploaded_file:
        # Resize & preprocessing
        ori_img = cv2.imdecode(np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        ori_img = cv2.cvtColor(cv2.resize(ori_img, (int(ori_img.shape[1] * scale), int(ori_img.shape[0] * scale))), cv2.COLOR_BGR2RGB)
        image = fn_image_transform((ori_img, 640 * 2 / sum(ori_img.shape))).to(device).unsqueeze(0).half()
        # Inference
        BiSeNetV2 = torch.jit.load("weights/BiSeNetV2.pt").to(device).half().eval()
        with torch.no_grad():
            mask = (torch.sigmoid(BiSeNetV2(image)[0, 0]).cpu().numpy() > threshold).astype("uint8") + 2
        # Caching
        st.session_state.image = cv2.resize(ori_img, (image.shape[3], image.shape[2]))
        st.session_state.mask = grab_cut(st.session_state.image, mask).astype("bool")

        del BiSeNetV2
        del image

    torch.cuda.empty_cache()
    gc.collect()

    # Grab cut
    mode = st.sidebar.radio("Grab cut mode", ["Include", "Exclude"], index=0, on_change=on_change_cb)
    mask = st.session_state.mask
    if not st.session_state.on_change:
        if st.session_state.coord is not None:
            coord = st.session_state.coord
            mask = cv2.circle(mask.astype("uint8") + 2, (coord["x"], coord["y"]), 32, 1 if mode == "Include" else 0, -1)
            mask = grab_cut(st.session_state.image, mask).astype("bool")
    st.session_state.on_change = False

    # Masking
    cat_image = st.session_state.image.copy()
    cat_image[mask] = 0.5 * cat_image[mask] + 0.5 * np.array([255, 0, 0])
    st.session_state.mask = mask
    streamlit_image_coordinates(cat_image, key="coord")

    # HierarchicalProbUNet
    st.button("Inferece", on_click=click_button)
    bs = st.sidebar.slider("Batch size", 1, 64, 32)
    mc_n = st.sidebar.slider("sampling number", 1, 30, 15)
    if st.session_state.inference:
        if "preds" not in st.session_state:
            HierarchicalProbUNet = torch.jit.load("weights/HierarchicalProbUNet.pt").to(device).half().eval()
            # Resize
            mask = zoom(mask.astype("float"), [a / b for a, b in zip(ori_img.shape[:2], mask.shape)]) > 0.5
            xmin, xmax = np.nonzero(mask.sum(0))[0][[0, -1]]
            ymin, ymax = np.nonzero(mask.sum(1))[0][[0, -1]]
            x_pad = ((xmax - xmin + (256 - step)) // step + 1) * step + (256 - step) - (xmax - xmin)
            y_pad = ((ymax - ymin + (256 - step)) // step + 1) * step + (256 - step) - (ymax - ymin)
            ori_img = ori_img[ymin:ymax, xmin:xmax]
            mask = mask[ymin:ymax, xmin:xmax]
            st.session_state.ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
            img = ori_img.copy()
            img[mask, :] = 0
            img = np.pad(ori_img, (((256 - step), y_pad - (256 - step)), ((256 - step), x_pad - (256 - step)), (0, 0)))
            mask = torch.from_numpy(mask[np.newaxis, :, :, np.newaxis]).to(device)

            transforms = A.Compose([A.Normalize(mean=(0.4379, 0.5198, 0.6954), std=(0.1190, 0.1178, 0.1243))])

            patches = torch.HalfTensor(patchify(np.transpose(transforms(image=img)["image"], (2, 0, 1)), (3, 256, 256), step=step)[0]).to(
                device
            )

            # Sort out patches to inference
            nx = patches.shape[1]
            patches = patches.reshape(-1, patches.shape[2], patches.shape[3], patches.shape[4])
            valid = []
            for i in range(patches.shape[0]):
                if not torch.all(patches[i] == 0):
                    valid.append(i)

            # Assign severity and lesion area
            # N x H x W x C
            preds = torch.zeros((mc_n, img.shape[0], img.shape[1], num_classes), dtype=torch.half).to(device)
            # N x H x W
            lesion_area = torch.zeros(preds.shape[:-1] + (1,), dtype=torch.half).to(device)

            # window
            window = torch.ones(256)
            window[: 256 - step] = torch.arange(256 - step) / (256 - step)
            window = torch.tile(window.unsqueeze(1), (1, 256))
            window = (
                (window * torch.rot90(window) * torch.rot90(window, 2) * torch.rot90(window, 3))
                .reshape((1,) + window.shape + (1,))
                .to(device)
            )

            # Inference with patch batch
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0.0, text=progress_text)
            end = len(valid) // bs + 1

            for i in range(end):
                my_bar.progress((i + 1) / end, text=progress_text)
                mini_patches = patches[valid[i * bs : min((i + 1) * bs, len(valid))]]
                area, logits = HierarchicalProbUNet.sample(img=mini_patches, mc_n=mc_n)
                for j in range(mini_patches.shape[0]):
                    y = (valid[i * bs + j] // nx) * step
                    x = (valid[i * bs + j] % nx) * step
                    preds[:, y : y + 256, x : x + 256] += logits[:, j] * window
                    lesion_area[:, y : y + 256, x : x + 256] += area[:, j] * window
            my_bar.empty()

            st.session_state.preds = preds[:, (256 - step) : (256 - step) + mask.shape[1], (256 - step) : (256 - step) + mask.shape[2]]
            st.session_state.lesion_area = (
                lesion_area[:, (256 - step) : (256 - step) + mask.shape[1], (256 - step) : (256 - step) + mask.shape[2]] * mask
            )
            st.session_state.mask_area = mask.sum().cpu()

            del HierarchicalProbUNet
            del patches
            del preds
            del lesion_area
            del mask
            del window

        torch.cuda.empty_cache()
        gc.collect()

        # Post processing
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(st.session_state.ori_img, cv2.COLOR_BGR2RGB))
            area_th = st.slider("area threshold", 0.0, 1.0, 0.5)
            e_bias = st.slider("Erythema", -1.0, 1.0, 0.0)
            i_bias = st.slider("Papulation", -1.0, 1.0, 0.0)
            ex_bias = st.slider("Excoriation", -1.0, 1.0, 0.0)
            l_bias = st.slider("Lichenification", -1.0, 1.0, 0.0)
            bias = torch.HalfTensor([[e_bias, i_bias, ex_bias, l_bias]]).to(device)

            lesion_area = st.session_state.lesion_area >= area_th
            areas = lesion_area.sum((1, 2, 3)) / st.session_state.mask_area  # N
            f"Area: {areas.mean() * 100:.1f}%(+/-{areas.std() * 100:.1f})"

        preds = (
            torch.cat(log_cumulative(cutpoints, bias + st.session_state.preds.reshape(-1, st.session_state.preds.shape[-1])), dim=-1)
            .argmax(-1)
            .reshape(st.session_state.preds.shape)
        ) * lesion_area
        
        severities = torch.div(preds.sum((1, 2)), lesion_area.sum((1, 2)))  # N x C
        
        easi = area2score(areas) * severities.sum(1) # N
        preds = (F.one_hot(preds, num_classes=4).half().mean(0)).permute(2, 0, 1, 3)  # C x H x W x 4
        preds = heatmap(st.session_state.ori_img, preds)

        torch.cuda.empty_cache()
        gc.collect()

        # Visualization
        with col2:
            st.image(cv2.cvtColor(preds[0], cv2.COLOR_BGR2RGB))
            f"Erythema: {severities[:, 0].mean():.2f}(+/-{severities[:, 0].std():.2f})"
            st.image(cv2.cvtColor(preds[1], cv2.COLOR_BGR2RGB))
            f"Papulation: {severities[:, 1].mean():.2f}(+/-{severities[:, 1].std():.2f})"
        with col3:
            st.image(cv2.cvtColor(preds[2], cv2.COLOR_BGR2RGB))
            f"Excoriation: {severities[:, 2].mean():.2f}(+/-{severities[:, 2].std():.2f})"
            st.image(cv2.cvtColor(preds[3], cv2.COLOR_BGR2RGB))
            f"Lichenification: {severities[:, 3].mean():.2f}(+/-{severities[:, 3].std():.2f})"
        with col1:
            f"EASI: {easi.mean():.1f}(+/-{easi.std():.1f})"
