import glob


import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


base_file_dir = './moss1_output/'
ano_file_dir = base_file_dir + 'mask_crop_data/'
img_file = base_file_dir + 'input.png'

ano_file_list = glob.glob(ano_file_dir + '*.png')

