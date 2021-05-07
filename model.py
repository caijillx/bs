import torchvision
from torch import nn
from config import *
from faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
import torch
from aod_model import AODnet
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from faster_rcnn.backbone.vgg_model import vgg
from faster_rcnn.network_files.rpn_function import AnchorsGenerator


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
        weights_dict["backbone.body.conv1.weight"] = weights_dict["backbone.body.conv1.weight"].repeat(1, 2, 1, 1)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        dh_model = AODnet()
        dh_model.load_state_dict(torch.load(pretrained_aod_model_path, map_location=device))
        print("正在生成联合模型......")
        return ODHModel(model, dh_model)
    elif type == "resnet":
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        backbone = resnet50_fpn_backbone()
        model = FasterRCNN(backbone=backbone, num_classes=91, image_mean=image_mean[0:3], image_std=image_std[0:3])
        weights_dict = torch.load(pretrained_res50_model_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("正在生成resnet50+fpn模型......")
        return model
    elif type == "vgg":
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        backbone = vgg(weights_path="/Users/llx/Downloads/vgg16.pth").features
        backbone.out_channels = 512
        anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                        output_size=[7, 7],  # roi_pooling输出特征矩阵尺寸
                                                        sampling_ratio=2)  # 采样率
        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler, image_mean=image_mean[0:3], image_std=image_std[0:3])
        print("正在生成vgg16模型......")
        return model
    else:
        raise Exception("请输入vgg、ohd、resnet种的一种")


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
