import torch
from torch import nn
import torch.nn.functional as F
import yaml
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.ao.quantization import get_default_qat_qconfig

class M5(nn.Module):
    def __init__(self, n_input, n_output, stride, n_channel, conv_kernel_sizes):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_input, n_channel, 
                               kernel_size=conv_kernel_sizes[0], 
                               stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, 
                               kernel_size=conv_kernel_sizes[1])
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, 
                               kernel_size=conv_kernel_sizes[2])
        self.bn3 = nn.BatchNorm1d(2*n_channel)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, 
                               kernel_size=conv_kernel_sizes[3])
        self.bn4 = nn.BatchNorm1d(2*n_channel)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x)) 
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(self.bn3(x))
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.relu4(self.bn4(x))
        x = self.pool4(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class QATM5(M5):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.qconfig = get_default_qat_qconfig('x86')

        # QAT linear
        self.fc1 = torch.ao.nn.qat.Linear(
            self.fc1.in_features, 
            self.fc1.out_features,
            qconfig=self.qconfig
        )
        
    def forward(self, x):
        x = self.quant(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)

        x = self.dequant(x)
        return F.log_softmax(x, dim=2)
    
    def fuse_model(self):
        # Fuse Conv+BN+ReLU modules
        self.eval()
        fuse_modules(self, [['conv1', 'bn1', 'relu1']], inplace=True)
        fuse_modules(self, [['conv2', 'bn2', 'relu2']], inplace=True)
        fuse_modules(self, [['conv3', 'bn3', 'relu3']], inplace=True)
        fuse_modules(self, [['conv4', 'bn4', 'relu4']], inplace=True)

