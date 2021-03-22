import torch
from torch import nn


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
