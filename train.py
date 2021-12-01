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


# load and preprocess data
def loader(train_path, test_path, batch_size):
    trans1 = transforms.Compose([transforms.Grayscale(), transforms.Lambda(lambda x: cv2.Canny(np.array(x), 150, 300)),
                                 transforms.ToPILImage(), transforms.ToTensor()])  ###
    train_imgs = datasets.ImageFolder(train_path, transform=trans1)
    train_iter = torch.utils.data.DataLoader(train_imgs, batch_size=batch_size, shuffle=True)

    trans2 = transforms.Compose([transforms.Grayscale(), transforms.Resize((32, 32)), transforms.ToTensor()])
    test_imgs = datasets.ImageFolder(test_path, transform=trans2)
    test_iter = torch.utils.data.DataLoader(test_imgs, batch_size=batch_size, shuffle=False)

    return train_iter, test_iter


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


def train_plot(fea, pre, cla, train_iter, test_iter, num_epochs, eta, device=try_gpu()):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)

    fea.apply(init_weights)
    pre.apply(init_weights)
    cla.apply(init_weights)
    print('training on', device)
    fea.to(device)
    pre.to(device)
    cla.to(device)

    func1 = nn.CrossEntropyLoss()
    func2 = nn.BCEWithLogitsLoss()
    trainer_fea = torch.optim.Adam(fea.parameters(), lr=0.001)  ###
    trainer_pre = torch.optim.Adam(pre.parameters(), lr=0.001)
    trainer_cla = torch.optim.Adam(cla.parameters(), lr=0.002)

    losses_pre = []
    losses_cla = []

    for epoch in range(num_epochs):
        loss_pre = 0
        loss_cla = 0
        for i, ((x_train, y), (x_test, _)) in enumerate(zip(train_iter, test_iter)):
            fea.train()
            pre.train()
            cla.train()
            x_train, x_test, y = x_train.to(device), x_test.to(device), y.to(device)

            mixed_data = torch.cat([x_train, x_test], dim=0)
            domain_label = torch.zeros([x_train.shape[0] + x_test.shape[0], 1]).cuda()
            domain_label[:x_train.shape[0]] = 1
            feature = fea(mixed_data)
            domain_logits = cla(feature.detach())
            loss_cla = func2(domain_logits, domain_label)
            loss_cla.backward()
            trainer_cla.step()

            class_logits = pre(feature[:x_train.shape[0]])
            domain_logits = cla(feature)
            loss_pre = func1(class_logits, y) - eta * func2(domain_logits, domain_label)
            loss_pre.backward()
            trainer_fea.step()
            trainer_pre.step()

            trainer_fea.zero_grad()
            trainer_pre.zero_grad()
            trainer_cla.zero_grad()

        losses_pre.append(loss_pre)
        losses_cla.append(loss_cla)

        print(
            "Epoch: {}/{}, label predictor loss: {}, domain classifier loss: {}".format(epoch + 1, num_epochs, loss_pre,
                                                                                        loss_cla))

    plt.plot(losses_pre, label="Label predictor loss")
    plt.plot(losses_cla, label="Domain classifer loss")
    plt.legend(loc='best')
    plt.savefig("Result.jpg")
    plt.show()
    torch.save(fea.state_dict(), "Feature-extractor.param")
    torch.save(pre.state_dict(), "Label-predictor.param")
    torch.save(cla.state_dict(), "Domain-classifier.param")


def feature_plot(train_iter, test_iter, fea):
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(train_iter, test_iter)):
        source_data = source_data.to(try_gpu())
        target_data = target_data.to(try_gpu())
        res1 = fea(source_data).detach().cpu()
        res2 = fea(target_data).detach().cpu()
        if i == 0:
            x1 = res1
            x2 = res2
        elif i > 4:
            break
        else:
            x1 = torch.cat((x1, res1))
            x2 = torch.cat((x2, res2))

    X = torch.cat((x1, x2))
    out = TSNE(n_components=2).fit_transform(X)
    p1 = out.T[0]
    p2 = out.T[1]
    plt.figure(figsize=(10, 10))
    plt.scatter(p1[:1000], p2[:1000], label="source")
    plt.scatter(p1[1000:], p2[1000:], label="target")
    plt.legend(loc='best')
    plt.savefig("Feature.jpg")
    plt.show()


if __name__ == "__main__":
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    epochs, batch_size, eta = 150, 500, 0.1 ###

    train_iter, test_iter = loader(train_path, test_path, batch_size)

    fea = feature_extractor().to(try_gpu())
    pre = label_predictor().to(try_gpu())
    cla = domain_classifier().to(try_gpu())
    train_plot(fea, pre, cla, train_iter, test_iter, epochs, eta)

    fea.load_state_dict(torch.load("Feature-extractor.param"))
    pre.load_state_dict(torch.load("Label-predictor.param"))
    cla.load_state_dict(torch.load("Domain-classifier.param"))
    feature_plot(train_iter, test_iter, fea)
