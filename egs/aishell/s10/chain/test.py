import logging
import os
import sys

import torch
#import torch.optim as optim
#from torch.nn.utils import clip_grad_value_
#from torch.optim.lr_scheduler import MultiStepLR
#from torch.utils.tensorboard import SummaryWriter
import kaldi
from kaldi_pybind import *
from options import get_args
def main():
    args = get_args()
    print(args)
if __name__ == '__main__':
    # torch.manual_seed(20191227)
    main()
