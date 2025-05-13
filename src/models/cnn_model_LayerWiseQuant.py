import torch
from torch import nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.ao.quantization import get_default_qat_qconfig

# from src.models.cnn_model import M5, QATM5, PTQM5

class M5Block(nn.Module):
    def __init__(self, conv, bn, relu, pool):
        super().__init__()
        self.block = nn.Sequential(conv, bn, relu, pool)

    def forward(self, x):
        return self.block(x)

class M5Modular(nn.Module):
    def __init__(self, n_input, n_output, stride, n_channel, conv_kernel_sizes):
        super().__init__()
        self.block1 = M5Block(
            nn.Conv1d(n_input, n_channel, kernel_size=conv_kernel_sizes[0], stride=stride),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.block2 = M5Block(
            nn.Conv1d(n_channel, n_channel, kernel_size=conv_kernel_sizes[1]),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.block3 = M5Block(
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=conv_kernel_sizes[2]),
            nn.BatchNorm1d(2*n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.block4 = M5Block(
            nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=conv_kernel_sizes[3]),
            nn.BatchNorm1d(2*n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4))
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class QATM5(M5Modular):
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

class PTQM5Modular(M5Modular):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.dequant(x)

        return F.log_softmax(x, dim=2)

    def fuse_model(self):
        # Fuse Conv+BN+ReLU modules
        self.eval()
        for block in [self.block1, self.block2, self.block3, self.block4]:
            torch.quantization.fuse_modules(block.block, ['0', '1', '2'], inplace=True)

class PTQM5_LayerWiseQuant(PTQM5Modular):
    def __init__(self, quantized_block_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantized_block_idx = quantized_block_idx

    def forward(self, x):
        for i in range(1, 5):
            if self.quantized_block_idx == i:
                x = self.quant(x)
                x = getattr(self, f'block{i}')(x)
                x = self.dequant(x)
            else:
                x = getattr(self, f'block{i}')(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

    # def fuse_model(self):
    #     # Only Fuse Conv+BN+ReLU modules for quantized block
    #     self.eval()
    #     for i in range(1, 5):
    #         if self.quantized_block_idx != i:
    #             torch.quantization.fuse_modules(getattr(self, f'block{i}').block, ['0', '1', '2'], inplace=True)

    def set_qconfig_for_layerwise(self, qconfig):
        for i in range(1, 5):
            block = getattr(self, f'block{i}')
            if i == self.quantized_block_idx:
                block.qconfig = qconfig
            else:
                block.qconfig = None
        self.fc1.qconfig = None  # Might change later.
        self.quant.qconfig = qconfig
        self.dequant.qconfig = qconfig
