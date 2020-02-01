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
            nn.Conv2d(20,40,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.rnn = nn.LSTMCell(36,112)
        self.fc = nn.Linear(112,3)
        self.h0 = torch.randn(112)
        self.c0 = torch.randn(112)

    def forward(self,imgs):
        h,c = self.h0, self.c0
        for img in imgs:
            feature = self.feature(img)
            feature = feature.view(-1,36)
            h,c = self.rnn(img,(h,c))
        output = torch.softmax(self.fc(h))
        return output

if __name__ == '__main__':
    model = RecogNet()
    print(model)