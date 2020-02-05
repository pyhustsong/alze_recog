'''
@Author: Ding Song
@Date: 2020-02-04 22:13:07
@LastEditors: Ding Song
@LastEditTime: 2020-02-04 22:13:07
@Description: 
'''
import os
import torch
import torch.nn as nn
from dataset import TrainData
from torch.utils.data import DataLoader
from model import RecogNet
from torch.optim import SGD

class AvgMea(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.total = 0
        self.right = 0

    def append(self,pred,label):
        self.total += len(label)
        self.right += torch.sum(pred==label)

    def cal(self):
        return self.right / self.total

class Solver(object):

    def __init__(self,learning_rate,num_epoch,save_dir):

        self.learning_rate = learning_rate
        self.num_epoch = num_epoch
        self.save_dir = save_dir
        self.model = RecogNet(3)
        self.avgmea = AvgMea()
        self.best_model = self.model
        self.optimizer = SGD(self.model.parameters(),lr=learning_rate)
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
                self.avgmea.append(pred,label)
                loss = self.loss(prediction,label)
                if (idx + 1) % 10 == 0:
                    print(f"[{epoch+1} | {self.num_epoch}] loss: {loss} acc: {self.avgmea.cal()}")
                loss.backward()
                self.optimizer.step()
            self.adjust_lr(epoch)

    def adjust_lr(self,epoch,decay=0.2):
        if (epoch + 1) % 5 == 0:
            for param_group in self.model.param_groups:
                param_group['lr'] *= decay

    def save_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir,'best_model.pth')
        torch.save(self.best_model.state_dict(),save_path)

def main():
    learning_rate = 5e-3
    num_epoch = 20
    save_dir = 'models'
    h5file = ''
    csvfile = ''
    dataset = TrainData(h5file,csvfile)
    train_loader = DataLoader(dataset,batch_size=5,shuffle=True)
    solver = Solver(learning_rate,num_epoch,save_dir)
    solver.train(train_loader)