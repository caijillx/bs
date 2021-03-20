import os
import torch
from torch import nn
from faster_rcnn import transforms
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from faster_rcnn.my_dataset import VOC2012DataSet
import torch.utils.data as Data
from faster_rcnn.train_utils import train_eval_utils as utils
from aod_model import AODnet


# ODH = object_detection+dehaze_image


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


def create_model(num_classes, device):
    backbone = resnet50_fpn_backbone()
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    od_model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    weights_dict = torch.load("faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location=device)
    weights_dict["backbone.body.conv1.weight"] = weights_dict["backbone.body.conv1.weight"].repeat(1, 2, 1, 1)
    missing_keys, unexpected_keys = od_model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = od_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    od_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    dh_model = AODnet()
    dh_model.load_state_dict(torch.load('aod_model_state_dict.pth', map_location=device))

    return ODHModel(od_model, dh_model)


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.RandomCrop(0.5)],),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    train_dataset = VOC2012DataSet(VOC_root, data_transform["train"], "train.txt")
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using %g dataloader workers' % nw)

    train_dataloader = Data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw, collate_fn=train_dataset.collate_fn
    )

    val_dataset = VOC2012DataSet(VOC_root, data_transform["val"], "val.txt")
    val_dataloader = Data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw, collate_fn=train_dataset.collate_fn
    )

    model = create_model(num_classes=6, device=device)

    model.to(device)

    params_od = [p for p in model.parameters() if p.requires_grad]

    for name, param in model.named_parameters():
        print(name, param.shape)

    optimizer = torch.optim.SGD(
        [{"params": model.dh_model.parameters(), "lr": 0.001},
         {"params": model.od_model.parameters(), "lr": 0.005}
         ]
        , lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.33
    )

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_mAP = []

    for epoch in range(parser_data.epochs):
        utils.train_one_epoch(model, optimizer, train_dataloader, device, epoch, train_loss=train_loss
                              , train_lr=learning_rate, print_freq=50, warmup=True)

        lr_scheduler.step()

        utils.evaluate(model, val_dataloader, device=device, mAP_list=val_mAP)

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from faster_rcnn.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_mAP) != 0:
        from faster_rcnn.plot_curve import plot_map
        plot_map(val_mAP)


if __name__ == "__main__":
    version = torch.version.__version__[:5]  # example: 1.6.0
    # 因为使用的官方的混合精度训练是1.6.0后才支持的，所以必须大于等于1.6.0
    if version < "1.6.0":
        raise EnvironmentError("pytorch version must be 1.6.0 or above")

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='/Users/llx/Desktop/RTTS', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=2, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
