import torch
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils import data
from torch import nn
import time
from IPython import display
import seaborn as sns
import numpy as np
import random
from torch.nn import functional as F
import torchvision
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import cv2
import sys


# construct net
class feature_extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
                                    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
                                    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.stage5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
                                    nn.MaxPool2d(kernel_size=2), nn.Dropout(p=0.5))

        self.conv = nn.Sequential(self.stage1, self.stage2, self.stage3, self.stage4, self.stage5)

    def forward(self, x):
        return self.conv(x).squeeze()


class label_predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 1024), nn.ReLU(),
                                    nn.Linear(1024, 9))  ###

    def forward(self, src):
        return self.linear(src)


class domain_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
                                    nn.Linear(512, 1))  ###

    def forward(self, x):
        return self.linear(x)


# train
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# predict
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
def predict(fea, pre, csv_path, device=try_gpu()):
    fea.eval()
    pre.eval()
    fea.to(device)
    pre.to(device)
    test = pd.read_csv(csv_path)
    result = []
    for _, (x_test, _) in enumerate(test_iter):
        x_test = x_test.to(try_gpu())
        y_pred = pre(fea(x_test))
        result.extend(argmax(y_pred, axis=1).cpu().numpy().tolist())

    index = [i for i in range(0, len(test))]
    output = pd.DataFrame({'ID': index, 'label': result})
    output.to_csv('Answer222.csv', index=None, encoding='utf8')


if __name__ == "__main__":
    # load and preprocess data
    lr, epochs, batch_size, eta = 0.001, 200, 500, 0.1
    csv_path = sys.argv[1]
    test_path = sys.argv[2]
    param1 = sys.argv[3]
    param2 = sys.argv[4]
    param3 = sys.argv[5]

    trans2 = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor()])
    test_imgs = datasets.ImageFolder(test_path, transform=trans2)
    test_iter = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=False)


    fea = feature_extractor().cuda()
    fea.load_state_dict(torch.load(param1))
    pre = label_predictor().cuda()
    pre.load_state_dict(torch.load(param2))
    cla = domain_classifier().cuda()
    cla.load_state_dict(torch.load(param3))

    predict(fea, pre, csv_path)
