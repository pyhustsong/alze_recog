import torch
import os
from model import RecogNet
from dataset import TestData
from torch.utils.data import DataLoader

class Test(object):

    def __init__(self,save_dir,num_classes):
        
        self.save_dir = save_dir
        self.model = RecogNet(num_classes)
        self._makedirs()

    def _makedirs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.sve_dir)

    def test(self,data_loader,model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.cuda()
        result = []
        for idx, img in enumerate(data_loader):
            img = img.cuda()
            prediction = self.model(img)
            pred = torch.max(prediction,dim=-1)[1]
            pred = pred.numpy().tolist()
            result.extend(pred)
        with open(os.path.join(self.save_dir,'result.csv'),'w') as f:
            for idx, r in enumerate(result):
                f.write(str(idx)+','+str(r)+'\n')

def main():
    save_dir = ''
    num_classes = 3
    h5file = ''
    model_path = ''
    testdata = TestData(h5file,img_size=48)
    test_loader = DataLoader(testdata,batch_size=10,shuffle=False)
    test = Test(save_dir,num_classes)
    test.test(test_loader,model_path)

if __name__ == '__main__':
    main()