'''
@Author: Ding Song
@Date: 2020-02-04 22:13:07
@LastEditors: Ding Song
@LastEditTime: 2020-02-04 22:13:07
@Description: 
'''
import torch
import torch.nn as nn
from dataset import TrainData
from model import RecogNet
from torch.optim import SGD

class Solver(object):

    def __init__(self,learning_rate,num_epoch):

        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.model = RecogNet()
        self.optimizer = SGD(self.model.parameters(),learning_rate=learning_rate)
        self.loss = nn.CrossEntropyLoss()

    def train(self,train_loader):
        self.model.train()
        self.model.cuda()
        for epoch in range(self.num_epoch):
            for idx, (img,label) in enumerate(train_loader):
                self.optimizer.zero_grad()
                img,label = img.cuda(),label.cuda()
                prediction = self.model(img)
                pred = prediction.max(dim=-1)
                loss = self.loss(prediction,label)
                loss.backward()
                self.optimizer.step()

    def adjust_lr(self,decay=0.2):
        for param_group in self.model.param_groups:
            param_group['lr'] *= decay