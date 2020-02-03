import torch
import torch.nn as nn

class RecogNet(nn.Module):

    def __init__(self):
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
        self.rnn = nn.LSTMCell(36*20,36*20)
        self.fc = nn.Linear(36*20,3)
        self.h0 = torch.randn(112).cuda()
        self.c0 = torch.randn(112).cuda()

    def forward(self,imgs):
        h,c = self.h0, self.c0
        channel = imgs.shape[1]
        for i in range(channel):
            feature = self.feature(imgs[:,i,...].unsqueeze(dim=1))
            print(feature.shape)
            feature = feature.view(36*40,-1)
            h,c = self.rnn(feature,(h,c))
        output = torch.softmax(self.fc(h))
        return output

if __name__ == '__main__':
    model = RecogNet()
    model.cuda()
    print(model)
    imgs = torch.randn(2,3,48,48).cuda()
    print(model(imgs))
