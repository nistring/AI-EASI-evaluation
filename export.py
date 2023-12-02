import torch
from SemanticSegmentation.semantic_segmentation import models
from SemanticSegmentation.semantic_segmentation import load_model
import sys
from main import LitModel
from model.model import HierarchicalProbUNet
from utils import *
import lightning.pytorch as pl

# sys.path.append("SemanticSegmentation")

# model_type = "BiSeNetV2"
# weight = "SemanticSegmentation/pretrained/model_segmentation_realtime_skin_30.pth"
# model = torch.load(weight, map_location=torch.device("cuda:0"))
# model = torch.jit.script(load_model(models[model_type], model).half().cuda())
# model.save(f"weights/{model_type}.pt")

########################

cfg = load_config("config.yaml")
litmodel = LitModel.load_from_checkpoint(
    "weights/lr_e-5_exdecay_0.99/checkpoints/epoch=486-step=6818.ckpt",
    model=HierarchicalProbUNet(
        latent_dims=cfg.model.latent_dims,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        loss_kwargs=dict(cfg.train.loss_kwargs),
        num_cuts=cfg.model.num_cuts,
    ),
    cfg=cfg,
    exp_name=None,
)
model = litmodel.model
model.half().eval().cuda()
print(model.cutpoints * model._alpha)
torch.jit.save(torch.jit.script(model), "weights/log_cumulative.pt")
