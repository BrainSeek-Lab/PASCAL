"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
from matplotlib.pyplot import xlim
import torch.nn as nn
from models.layer import *

counter = 0
#mylist = [8 for i in range(50)]
#mylist = [4,2,4,1,4,2,2,1,4,2,2,1,4,1,1,1,4]
mylist = [4,4,4,1,4,4,4,1,4,1,1,1,1,1,1,1,4]
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        global counter	
        global mylist
        #residual function
        self.T = mylist[counter-1] 
        self.T_new = mylist[counter]
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            IF(counter, mylist),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        counter = counter + 1
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.act = IF(counter,mylist)
        counter = counter + 1

    def forward(self, x):
        x1 = self.residual_function(x) 
        x2 = self.shortcut(x)
        if(self.T == self.T_new):
            x = x1 + x2
            return self.act(x)
        else:
            y_shape = [self.T, int(x2.shape[0]/self.T)]
            y_shape.extend(x2.shape[1:])
            x2_new = x2.view(y_shape)
            acc = torch.zeros(x2_new[0].shape, device = 0)
            for t in range(self.T):
                acc = acc + x2_new[t]
            
            y_shape = [self.T_new, int(x1.shape[0]/self.T_new)]
            y_shape.extend(x1.shape[1:])
            x1_new = x1.view(y_shape)
            x1_new[0] = x1_new[0] + acc
            x1_new = x1_new.flatten(0,1).contiguous()
            return self.act(x1_new)


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        global counter
        global mylist
        self.in_channels = 64
        self.is_training = False
        self.merge = MergeTemporalDim(mylist[-1])
        self.expand = ExpandTemporalDim(mylist[-1])
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            IF(counter, mylist))
        counter = counter + 1
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def set_train(self): 
        for module in self.modules():
            if isinstance(module, IF):
                module.T = 0
        self.is_training = True
        return
    
    def forward(self, x):
        if(self.is_training == True):
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
            output = self.fc(output)
            return output
        else:
            output = self.conv1(x)
            output = self.conv2_x(output)
            output = self.conv3_x(output)
            output = self.conv4_x(output)
            output = self.conv5_x(output)
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1) 
            output = self.fc(output)
            output = self.expand(output)
            return output

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
