import random
import torch

from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor
        convert PIL Image or numpy.ndarray(HxWxC)
        in the range [0,255] to a torch.FloatTensor of shape (CxHxW) in the range[0.0,1.0]
    """

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


class RandomCrop(object):
    """随机裁剪图片并重新定位bbox"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            bbox = target["boxes"]
            max_bbox = torch.cat([torch.min(bbox[:, 0:2], axis=0).values, torch.max(bbox[:, 2:4], axis=0).values],
                                 dim=0)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = width - max_bbox[2]
            max_d_trans = height - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(width, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(height, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[:, crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bbox[:, [0, 2]] = bbox[:, [0, 2]] - crop_xmin
            bbox[:, [1, 3]] = bbox[:, [1, 3]] - crop_ymin
            target["boxes"] = bbox
        return image, target
