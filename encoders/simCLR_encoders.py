import torch.nn as nn
import torchvision.models as models
import torch
from torch.nn.parallel import DistributedDataParallel


class SimCLREncoder(nn.Module):
    """
    ResNet18 encoder adapted for MNIST
    Args:
        forward:
            x (Tensor): batch of images
    Returns:
        model(x) (Tensor): batch of encoded images
    """

    def __init__(self, encoder_type, out_dim, device, DDP=True, local_rank=None, checkpoint=None):
        super(SimCLREncoder, self).__init__()
        self.model_dict = {'resnet18': models.resnet18(),
                           'resnet50': models.resnet50(),
                           'resnet101': models.resnet101()}
        self.encoder = self.model_dict[encoder_type]
        self.out_dim = out_dim
        self.checkpoint = checkpoint
        self.in_size = self._get_output_shape()
        self.projection_head = self._get_projection_head(self.in_size)
        self.model = self._get_model().to(device)
        if DDP:
            self.model = DistributedDataParallel(self.model, device_ids=[local_rank], output_device=local_rank)


    def _get_output_shape(self, image_dim=(1, 3, 224, 224)):
        model = nn.Sequential(*list(self.encoder.children())[:-1],
                      nn.Flatten())
        return model(torch.rand(*image_dim)).data.shape[1]

    def _get_model(self):
        if self.checkpoint is not None:
            model = torch.load(self.checkpoint)
        else:
            resnet = self.encoder
            model = nn.Sequential(*list(resnet.children())[:-1],
                                  nn.Flatten(),
                                  self.projection_head)
        return model

    def _get_projection_head(self, in_size):
        projection_head = nn.Sequential(nn.Linear(in_size, in_size),
                                        nn.ReLU(),
                                        nn.Linear(in_size, self.out_dim))
        return projection_head

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)

