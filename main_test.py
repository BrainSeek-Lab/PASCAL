import argparse
import os
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from models import modelpool
from preprocess import datapool
from utils import train, val, seed_all, get_logger
from models.layer import *
import time
import math

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument('-j','--workers',default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg16',type=str,help='model')
parser.add_argument('-id', '--identifier', type=str,help='model statedict identifier')

# test configuration
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')
parser.add_argument('-L', '--levels', default=8, type=int, help='number of levels')
parser.add_argument('-conventional_model', '--conventional_model', default="true", type=str, help="Load a model with model_state_dict")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def main():
    global args
    seed_all(args.seed)
    # preparing data
    train_loader,test_loader = datapool(args.dataset, args.batch_size)
    # preparing model
    model = modelpool(args.model, args.dataset)
    #print(model)
    model_dir = '%s-checkpoints'% (args.dataset)
    if(args.conventional_model == "true"):
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(os.path.join(model_dir, args.identifier + '.pth'), map_location=torch.device('cpu'))['model_state_dict']

    ########### LAYER SPECIFICATIONS #####################
    #levels = [8 for i in range(50)]	
    levels = [8,8,8]+[4,4,4]+[4,4,4]+[1,1,1]+[4,4,4]+[4,4,4]+[1,1,1]+[1,1,1]+[1,1,1]+[1,1,1]+[1,1,1]+[4,4,4]+[8]+[8]+[8] #for VGG
    #levels = [8 for i in range(100)]
    #levels = [4,4]+[4,4]+[4,4]+[1,1]+[4,4]+[4,4,4,4]+[4,4]+[1,1]+[4,4]+[1,1,4,4]+[1,1]+[1,1]+[1,1]+[1,1,1,1]+[1,1]+[1,1]+[4] 
    #levels = [4,4]+[2,2]+[4,4]+[1,1]+[4,4]+[2,2,4,4]+[2,2]+[1,1]+[4,4]+[2,2,4,4]+[2,2]+[1,1]+[4,4]+[1,1,4,4]+[1,1]+[1,1]+[4]
    #levels = [4 for i in range(39)]
    levels = torch.FloatTensor(levels).to(0)


    index = 0
    # if old version state_dict
    keys = list(state_dict.keys())
    keylist = []
    ############ FOR VGG16 #################################
    if(args.model[:3] == "vgg"): 
        for k in keys:
            if((k[-4:] == 'bias' or k[-12:] == 'running_mean') and (k!= 'layer1.0.weight') and (k!= 'layer1.0.bias') and (k!= 'layer1.1.running_mean') and (k!= 'layer1.1.bias')):
                state_dict[k] = torch.div(state_dict[k].double().to(0),levels[index]) #dividing by L to match the ANN
                index = index + 1
            if(k[-6:] == "thresh" or k[-2:] == "up"):
                keylist.append(k)
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
            #print(f"In VGG: Index = {index}")
    
    ########### FOR RESNET18 ################################
    else: 
        for k in keys:
            if((k[-4:] == 'bias' or k[-12:] == 'running_mean') and (k!= 'conv1.1.weight') and (k!= 'conv1.1.bias') and (k!= 'conv1.1.running_mean')):
                state_dict[k] = torch.div(state_dict[k].double().to(0),levels[index]) #dividing by L to match the ANN
                index = index + 1
            if(k[-6:] == "thresh" or k[-2:] == "up"):
                keylist.append(k)
            if "relu.up" in k:
                state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
            elif "up" in k:
                state_dict[k[:-2]+'thresh'] = state_dict.pop(k)
            #print(f"In ResNet: Index = {index}")
    
    model.load_state_dict(state_dict)

    model.to(device)

    acc = val(model, test_loader, device, 4) #T needs to be positive
    print(acc)



if __name__ == "__main__":
    main()
