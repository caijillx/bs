from torch import nn
from config import *
from faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
import torch
from aod_model import AODnet
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor


class ODHModel(nn.Module):

    def __init__(self, od_model, dh_model):
        super(ODHModel, self).__init__()
        self.dh_model = dh_model
        self.od_model = od_model

    def forward(self, images, targets=None):
        images = list(images)
        dh_images = [self.dh_model(image.unsqueeze(0)) for image in images]
        images = [torch.cat([image, dh_image.squeeze(0)], dim=0) for image, dh_image in zip(images, dh_images)]
        results = self.od_model(images, targets)
        return results


# 如果type为ohd，则创建AODNet+Faster_rcnn合体模型,否则则创建只有faster_rcnn的模型。
def create_model(num_classes, image_mean=None, image_std=None, device="cpu", type="ohd"):
    if type == "ohd":
        # 因为需要连接图片和去雾后的图片，需要两层，repeat参数改为True
        if image_mean is None:
            image_mean = [0.551, 0.548, 0.541, 0.384, 0.378, 0.372]
        if image_std is None:
            image_std = [0.193, 0.194, 0.198, 0.195, 0.210, 0.219]
        backbone = resnet50_fpn_backbone(repeat=True)
        model = FasterRCNN(backbone=backbone, num_classes=91, image_mean=image_mean, image_std=image_std)
        weights_dict = torch.load(pretrained_res50_model_path, map_location=device)
        # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
        weights_dict["backbone.body.conv1.weight"] = weights_dict["backbone.body.conv1.weight"].repeat(1, 2, 1, 1)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        dh_model = AODnet()
        dh_model.load_state_dict(torch.load(pretrained_aod_model_path, map_location=device))
        print("正在生成去雾+目标检测模型......")
        return ODHModel(model, dh_model)
    else:
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        backbone = resnet50_fpn_backbone()
        model = FasterRCNN(backbone=backbone, num_classes=91, image_mean=image_mean[0:3], image_std=image_std[0:3])
        weights_dict = torch.load(pretrained_res50_model_path, map_location=device)
        # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("正在生成目标检测模型......")
        return model


def create_optimizer(model, type="ohd"):
    if type == "ohd":
        optimizer = torch.optim.SGD(
            [{"params": model.dh_model.parameters(), "lr": 0.001},
             {"params": model.od_model.parameters(), "lr": 0.005}]
            , lr=0.005, momentum=0.9, weight_decay=0.0005
        )
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    return optimizer
