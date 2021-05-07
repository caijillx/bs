import os
import time
import json
import torch
from PIL import Image
import matplotlib.pyplot as plt
from train_model import ODHModel
from torchvision import transforms
from faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN
from faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from faster_rcnn.draw_box_utils import draw_box
from aod_model import AODnet


def create_model(num_classes):
    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone(repeat=True)
    od_model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    # AODnet
    dh_model = AODnet()

    return ODHModel(od_model, dh_model)


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=6)

    # load train weights
    train_weights = "/Users/llx/Downloads/AOD-model-3.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = 'faster_rcnn/pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    original_img = Image.open("/Users/llx/Desktop/dehaze_object_detection/test_images/test5.jpeg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time.time()
        predictions = model(img.to(device))[0]
        print("inference+NMS time: {}".format(time.time() - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img, predict_boxes, predict_classes, predict_scores, category_index,
                 thresh=0.5, line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        # 保存预测的图片结果
        original_img.save("result_images/test_result5.jpg")


if __name__ == '__main__':
    main()
