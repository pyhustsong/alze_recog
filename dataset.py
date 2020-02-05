'''
@Author: Ding Song
@Date: 2020-02-04 20:05:21
@LastEditors  : Ding Song
@LastEditTime : 2020-02-05 18:23:01
@Description: 
'''
import h5py
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np 
from PIL import Image 
from torch.utils.data import Dataset,DataLoader

class TrainData(Dataset):

    def __init__(self,h5file,csvfile,img_size):

        self.data = h5py.File(h5file,'r')['data']
        with open(csvfile,'r') as f:
            lines = f.readlines()[1:]
        self.label_file = lines
        self.img_size = img_size

    def __getitem__(self,idx):
        img = self.data[idx][0]
        img = img.transpose(1,0,2)
        new_img = np.zeros((img.shape[0],self.img_size,self.img_size),dtype=np.float32)
        for i in range(img.shape[0]):
            new_img[i] = cv2.resize(img[i],(self.img_size,self.img_size))
        label = int(self.label_file[idx].strip().split(',')[-1])
        return torch.from_numpy(new_img),torch.tensor(label)

    def __len__(self):
        return len(self.label_file) - 1

class TestData(Dataset):

    def __init__(self,h5file,img_size):

        self.data = h5py.File(h5file,'r')['data']
        self.img_size = img_size

    def __getitem__(self,idx):
        img = self.data[idx][0]
        img = img.transpose(1,0,2)
        new_img = np.zeros((img.shape[0],self.img_size,self.img_size),dtype=np.float32)
        for i in range(img.shape[0]):
            new_img[i] = cv2.resize(img[i],(self.img_size,self.img_size))
        return torch.from_numpy(new_img)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    h5file = '/home/song/workspace/datasets/recog-alzheimer/train/train_pre_data.h5'
    csvfile = '/home/song/workspace/datasets/recog-alzheimer/train/train_pre_label.csv'
    dataset = TrainData(h5file,csvfile)
    dataloader = DataLoader(dataset,num_workers=2,batch_size=2)
    for idx,(img,label) in enumerate(dataloader):
        print(idx)
        print(img.shape)
        print(label.tolist())