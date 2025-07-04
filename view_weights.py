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
parser.add_argument('-L', '--levels', default=4, type=int, help='number of levels')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    model_dir = '%s-checkpoints'% (args.dataset)
    final_path = os.path.join(model_dir, args.identifier + '.pth')
    model = torch.load(final_path, map_location=torch.device('cpu') )
    #print(model['model_state_dict'].keys())
    for key in model.keys():
        if(key[-1] == 't'):
            tensor = model[key]
            print(key,torch.mean(torch.abs(tensor.flatten())),tensor.flatten().shape)
if __name__ == "__main__":
    main()
