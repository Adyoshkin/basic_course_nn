import torch.nn as nn


def conv_bn_relu(in_planes, out_planes, kernel=3, stride=1):
    net = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(True))
    return net

def transpose_conv_bn_relu(in_planes, out_planes, kernel=3, stride=1):
    net = nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=1),
                        nn.BatchNorm2d(num_features=out_planes),
                        nn.ReLU(True))
    return net

# input size Bx3x224x224
class SegmenterModel(nn.Module):
    def __init__(self, in_size=3):
        super(SegmenterModel, self).__init__()
        
        # implement your model here
        D1 = 128
        D2 = 256
        D3 = 512
        x = 2
        
        # in_size*224*224
        
        self.conv_bn_relu1 = nn.Sequential()                    
        self.conv_bn_relu1.add_module('conv_bn_relu1_1', conv_bn_relu(in_size, D1))
        self.conv_bn_relu1.add_module('conv_bn_relu1_2', conv_bn_relu(D1, D1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # D1*112*112
        
        self.conv_bn_relu2 = nn.Sequential()
        self.conv_bn_relu2.add_module('conv_bn_relu2_1', conv_bn_relu(D1, D2))
        self.conv_bn_relu2.add_module('conv_bn_relu2_2', conv_bn_relu(D2, D2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # D2*56*56
        
        self.conv_bn_relu3 = nn.Sequential()
        self.conv_bn_relu3.add_module('conv_bn_relu3_1', conv_bn_relu(D2, D3))
        self.conv_bn_relu3.add_module('conv_bn_relu3_2', conv_bn_relu(D3, D3))
        
        # D3*56*56
        
        self.transpose_conv_bn_relu3 = nn.Sequential()
        self.transpose_conv_bn_relu3.add_module('transpose_conv_bn_relu3_1', transpose_conv_bn_relu(D3, D3))
        self.transpose_conv_bn_relu3.add_module('transpose_conv_bn_relu3_2', transpose_conv_bn_relu(D3, D2))
        
        # D2*56*56
        
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.transpose_conv_bn_relu2 = nn.Sequential()
        self.transpose_conv_bn_relu2.add_module('transpose_conv_bn_relu2_1', transpose_conv_bn_relu(D2, D2))
        self.transpose_conv_bn_relu2.add_module('transpose_conv_bn_relu2_2', transpose_conv_bn_relu(D2, D1))
        
        # D2*112*112
        
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.transpose_conv_bn_relu1 = nn.Sequential()
        self.transpose_conv_bn_relu1.add_module('transpose_conv_bn_relu1_1', transpose_conv_bn_relu(D1, D1)) 
        self.transpose_conv_bn_relu1.add_module('transpose_conv_bn_relu1_2', transpose_conv_bn_relu(D1, x))
        
        # x*224*224
        
    def forward(self, x):
        
        output = self.conv_bn_relu1(x)
        output, indices1 = self.pool1(output)
        
        output = self.conv_bn_relu2(output)
        output, indices2 = self.pool2(output)
        
        output = self.conv_bn_relu3(output)
        output = self.transpose_conv_bn_relu3(output)
        
        output = self.unpool2(output, indices2)
        output = self.transpose_conv_bn_relu2(output)
        
        output = self.unpool1(output, indices1)
        output = self.transpose_conv_bn_relu1(output)

        return output
