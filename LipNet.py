import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable

class LipSeqLoss(nn.Module):
 
    def __init__(self):
        super(LipSeqLoss, self).__init__()
        self.criterion = nn.NLLLoss(reduction='none')

    def forward(self, input, length, target):
        loss = []
        transposed = input.transpose(0, 1).contiguous()
        for i in range(transposed.size(0)):
            loss.append(self.criterion(transposed[i, ], target.squeeze(1)).unsqueeze(1))
        loss = torch.cat(loss, 1)
        
        #GPU version
        mask = torch.zeros(loss.size(0), loss.size(1)).float().cuda()
        # Cpu version
#         mask = torch.zeros(loss.size(0), loss.size(1)).float()   

        for i in range(length.size(0)):
            L = min(mask.size(1), length[i])
            mask[i, L-1] = 1.0
        loss = (loss * mask).sum() / mask.sum()
        return loss

class LipNet(torch.nn.Module):
    def __init__(self, init_features_num=64, drop_rate=0.3, type_class=313):
        super(LipNet, self).__init__()
        self.drop_rate = drop_rate
        self.type_class = type_class 

        # Cnn
        self.features = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(3, init_features_num, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)),
            ('norm', nn.BatchNorm3d(init_features_num)),
            ('relu', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))), ]))
     
        # Rnn
        self.gru1 = nn.GRU(64*28*28, 256, bidirectional=True, batch_first=True) 
        self.gru2 = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        # Fc
        self.fc = nn.Sequential(
            nn.Dropout(self.drop_rate),
            nn.Linear(512, self.type_class) )
        
        
    def forward(self, x):
        self.gru1.flatten_parameters()
        self.gru2.flatten_parameters()
        # Cnn
        cnn = self.features(x)
        cnn = cnn.permute(0, 2, 1, 3, 4).contiguous()
        batch, seq, channel, high, width = cnn.size()
        cnn = cnn.view(batch, seq, -1)
        # Rnn
        rnn, _ = self.gru1(cnn)
        rnn, _ = self.gru2(rnn)
        # Fc
        fc = self.fc(rnn).log_softmax(-1)
        return fc
    
    
