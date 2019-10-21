import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReadData(Dataset):

    def __init__(self, image_root, seq_max_lens=15):
        self.seq_max_lens = seq_max_lens
        self.data_root = image_root
        self.data = []


        # linux: /   windows:\\
        pic_file_path = [root for root, dirs, files in os.walk(self.data_root) if root.split('/')[-1]!=self.data_root.split('/')[-1]]
        file_names = [i.split('/')[-1] for i in pic_file_path]

        self.lengths = [len(os.listdir(path)) for path in pic_file_path]
        self.data = [(file_name, length,) for file_name, length in zip(file_names, self.lengths)]        
        self.data = list(filter(lambda sample: sample[-1] <= self.seq_max_lens, self.data)) 
   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        (path, pic_nums) = self.data[idx]
        path = os.path.join(self.data_root, path)
        files = [os.path.join(path, ('{}' + '.png').format(i)) for i in range(1, pic_nums+1)]
        files = filter(lambda path: os.path.exists(path), files)
        frames = [cv2.imread(file) for file in files ] 
        frames_ = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  for img in frames]       
        length = len(frames_)    
        channels=3
        picture_h_w=112
        vlm = torch.zeros((channels, self.seq_max_lens, picture_h_w, picture_h_w))
        for i in range(len(frames_)):
            result = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((picture_h_w, picture_h_w)),
                transforms.CenterCrop((picture_h_w, picture_h_w)),
                transforms.ToTensor(),
                transforms.Normalize([0, 0, 0], [1, 1, 1]) 
            ])(frames_[i])
            vlm[:, i] = result       
        return {'volume': vlm, 'length': length, 'key': path}
