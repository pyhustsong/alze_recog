'''
@Author: Ding Song
@Date: 2020-02-04 20:36:03
@LastEditors  : Ding Song
@LastEditTime : 2020-02-04 22:14:09
@Description: 
'''
import torch
import torch.nn as nn

class RecogNet(nn.Module):

    def __init__(self,num_classes):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1,10,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(10,20,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(20,20,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.rnn = nn.LSTMCell(20*36,20*36)
        self.fc = nn.Linear(36*20,num_classes)

    def forward(self,imgs):
        b,channel = imgs.shape[0],imgs.shape[1]
        h,c = torch.zeros(b,720).cuda(), torch.zeros(b,720).cuda()
        for i in range(channel):
            feature = self.feature(imgs[:,i,...].unsqueeze(dim=1))
            feature = feature.view(-1,36*20)
            h,c = self.rnn(feature,(h,c))
        output = torch.softmax(self.fc(h),dim=-1)
        return output

if __name__ == '__main__':
    model = RecogNet(3)
    model.cuda()
    print(model)
    imgs = torch.randn(2,5,48,48).cuda()
    print(model(imgs))
