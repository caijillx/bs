import os
import torch
from faster_rcnn import transforms
from faster_rcnn.my_dataset import VOC2012DataSet
import torch.utils.data as Data
from faster_rcnn.train_utils import train_eval_utils as utils
from model import create_model, create_optimizer


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5),
                                     transforms.RandomCrop(0.5)]),
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

    model = create_model(num_classes=6, device=device, type=parser_data.type)
    model.to(device)
    optimizer = create_optimizer(model, type=parser_data.type)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.66)

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
        # 调整学习率
        lr_scheduler.step()
        # 执行评测
        utils.evaluate(model, val_dataloader, device=device, mAP_list=val_mAP)

       # save weights
        save_files = {
            'od_model': model.state_dict(),
            'optimizer_od': optimizer.state_dict(),
            'lr_scheduler_od': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/{}-model-{}.pth".format(parser_data.type, epoch))

    # 画出loss和lr的曲线
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from faster_rcnn.plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # 画出mAP的曲线
    if len(val_mAP) != 0:
        from faster_rcnn.plot_curve import plot_map
        plot_map(val_mAP)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

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
    # 训练的模型类型
    parser.add_argument('--type', default="vgg", type=str, help='model type when training.')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
