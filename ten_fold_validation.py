import torch.utils.data as Data
from config import *
from validation import validate
from faster_rcnn import transforms
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN
from faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from faster_rcnn.my_dataset import VOC2012DataSet
import torch
from faster_rcnn.train_utils import train_eval_utils as utils
from aod_model import AODnet
from model import ODHModel
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("use device", device)

data_transform = {
    "train": transforms.Compose([transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(0.5),
                                 transforms.RandomCrop(0.5)], ),
    "val": transforms.Compose([transforms.ToTensor()]),
    "trainval": transforms.Compose([transforms.ToTensor()]),
}

transforms = transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                 transforms.RandomCrop(0.5)], )

dataset = VOC2012DataSet(VOC_root, data_transform["trainval"], "trainval.txt")

image_mean = [0.551, 0.548, 0.541, 0.384, 0.378, 0.372]
image_std = [0.193, 0.194, 0.198, 0.195, 0.210, 0.219]

k = 10


def get_k_fold_data(k, i, dataset):
    assert k > 1
    with open(VOC_root + "/ImageSets/Main/trainval.txt", 'r') as f:
        xml_list = f.readlines()
    print(len(xml_list))
    if len(dataset) % k == 0:
        cnt = len(dataset) // k
    else:
        cnt = len(dataset) // k + 1
    print("cnt", cnt)
    train_xmls, val_xmls = [], []
    for j in range(0, k):
        if j == i:
            val_xmls = xml_list[j * cnt:(j + 1) * cnt]
        else:
            train_xmls.extend(xml_list[j * cnt:(j + 1) * cnt])
    # 由于训练集需要在训练的时候更好的进行水平反转和随机剪切。
    train_dataset = VOC2012DataSet(VOC_root, data_transform["train"], "trainval.txt")
    train_dataset.union(train_xmls)
    val_dataset = VOC2012DataSet(VOC_root, data_transform["val"], "trainval.txt")
    val_dataset.union(val_xmls)
    return train_dataset, val_dataset


# 如果type为ohd，则创建AODNet+Faster_rcnn合体模型,否则则创建只有faster_rcnn的模型。
def create_model(num_classes, device="cpu", type="ohd"):
    if type == "ohd":
        # 因为需要连接图片和去雾后的图片，需要两层，repeat参数改为True
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


# IOU取0.5-0.95 step 0.05
coco_mAPs = []
# IOU取0.5
voc_mAPs = []
# 单类别IOU取0.5
voc_cat_mAPS = []

for i in range(k):
    print("正在测试第{}轮.......".format(i))
    train_dataset, val_dataset = get_k_fold_data(k, i, dataset)
    model = create_model(num_classes=num_classes)
    model.to(device)
    train_dataloader = Data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn
    )

    val_dataloader = Data.DataLoader(
        val_dataset, batch_size=2, shuffle=False, collate_fn=train_dataset.collate_fn
    )

    optimizer = torch.optim.SGD(
        [{"params": model.dh_model.parameters(), "lr": 0.001},
         {"params": model.od_model.parameters(), "lr": 0.005}
         ]
        , lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.66
    )

    train_loss = []
    learning_rate = []
    val_mAP = []

    epochs = 10

    for epoch in range(epochs):
        utils.train_one_epoch(model, optimizer, train_dataloader, device, epoch, train_loss=train_loss
                              , train_lr=learning_rate, print_freq=50, warmup=True)

        lr_scheduler.step()

    category_index = {1: 'person', 2: 'bus', 3: 'bicycle', 4: 'car', 5: 'motorbike'}

    coco_mAP, voc_mAP, voc_map_info_list = validate(model, val_dataset, val_dataloader, category_index=category_index,
                                                    device=device,
                                                    mAP_list=val_mAP)

    coco_mAPs.append(coco_mAP)
    voc_mAPs.append(voc_mAP)
    voc_cat_mAPS.append(voc_map_info_list)

print(coco_mAPs)

print(voc_mAPs)

print(voc_cat_mAPS)
