import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image 
from torch.utils.data import Dataset,DataLoader

class TrainData(Dataset):

    def __init__(self,h5file,csvfile):

        self.data = h5py.File(h5file,'r')['data']
        with open(csvfile,'r') as f:
            lines = f.readlines()[1:]
        self.label_file = lines

    def __getitem__(self,idx):
        img = self.data[idx][0]
        img = img.transpose(1,0,2)
        label = self.label_file[idx].strip()
        return torch.from_numpy(img),torch.Tensor(label)

if __name__ == '__main__':
    h5file = '../alze_recog/train_pre_data.h5'
    csvfile = '../alze_recog/train_pre_label.csv'
    dataset = TrainData(h5file,csvfile)
    dataloader = DataLoader(dataset,num_workers=2,batch_size=2)
    for idx,(img,label) in enumerate(dataloader):
        print(idx)
        print(img.shape)
        print(label.tolist())