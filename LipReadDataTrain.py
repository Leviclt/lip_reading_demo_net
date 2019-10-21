import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReadData(Dataset):

    def __init__(self, image_root, label_root, seq_max_lens):
        self.seq_max_lens = seq_max_lens
        self.data = []
        self.data_root = image_root
        with open(label_root, 'r', encoding='utf8') as f:
            lines = f.readlines()
            lines = [line.strip().split('\t') for line in lines]
            self.dictionary = sorted(np.unique([line[1] for line in lines])) 
            pic_path = [image_root + '/' + line[0] for line in lines] 
            self.lengths = [len(os.listdir(path)) for path in pic_path]
            
            save_dict = pd.DataFrame(self.dictionary, columns=['dict'])
            save_dict.to_csv('./dictionary/dictionary.csv', encoding='utf8', index=None)  #save dict

            self.data = [(line[0], self.dictionary.index(line[1]), length) for line, length in zip(lines, self.lengths)]
            self.data = list(filter(lambda sample: sample[-1] <= self.seq_max_lens, self.data))      


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (path, label, pic_nums) = self.data[idx]
        path = os.path.join(self.data_root, path)
        files = [os.path.join(path, ('{}' + '.png').format(i)) for i in range(1, pic_nums+1)]
        files = filter(lambda path: os.path.exists(path), files)
        frames = [cv2.imread(file) for file in files ] 
        frames_ = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in frames]       
        length = len(frames_)
        channels = 3
        picture_h_w = 112
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
        
        return {'volume': vlm, 'label': torch.LongTensor([label]), 'length': length}
  
