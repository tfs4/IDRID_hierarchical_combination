import torchvision.models as models
import torch
from torch import nn

import config
import random
import numpy as np

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)


def get_densenet121_3_classes():
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 3),
    )
    densenet121.cuda()

    return densenet121



def get_densenet201_2_classes():
    model = models.densenet169(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(

        nn.Linear(1664, 1024),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
        nn.Softmax(),
    )

    model.cuda()

    return model


# best result 83.4
def get_densenet121_2_classes():
    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2),
    )
    densenet121.cuda()

    return densenet121


def get_vgg_last_3():

    densenet121 = models.inception_v3(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(256, 3),
    )
    densenet121.cuda()

    return densenet121



def get_densenet121_last_3():

    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(256, 3),
    )
    densenet121.cuda()

    return densenet121


# Accuracy on Test set = 53.623188%
def get_densenet121_mc():

    densenet121 = models.densenet121(pretrained=True)

    for param in densenet121.parameters():
        param.requires_grad = False

    densenet121.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.4),

        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.4),


        nn.Linear(256, 4),
    )
    densenet121.cuda()

    return densenet121


