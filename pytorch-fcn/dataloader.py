import glob
import random

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


base_file_dir = './moss1_output/'
ano_file_dir = base_file_dir + 'mask_crop_data/'
img_file = 'input.png'

ano_file_list = glob.glob(ano_file_dir + '*.png')

eval_data_list = random.sample(ano_file_list, 10)
study_data_list = list(set(ano_file_list) - set(eval_data_list))

img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_height, img_width, img_channel = img.shape

study_imgs = np.zeros([len(study_data_list), img_height, img_width, img_channel])
eval_imgs = np.zeros([len(eval_data_list), img_height, img_width, img_channel])
ano_study_imgs = np.zeros([len(study_data_list), img_height, img_width, img_channel])
ano_eval_imgs = np.zeros([len(eval_data_list), img_height, img_width, img_channel])

for i, s in enumerate(study_data_list):
    study_imgs[i] = img
    ano_study_imgs[i] = cv2.imread(s)

# for i, e in enumerate(eval_data_list):
#     eval_imgs[i] = img
#     ano_eval_imgs[i] = cv2.imread(e)

study_imgs = torch.tensor(study_imgs, dtype = torch.float32)                 #ndarray - torch.tensor
ano_study_imgs = torch.tensor(ano_study_imgs, dtype = torch.float32)
study_dataset = TensorDataset(study_imgs, ano_study_imgs)
study_dataloader = DataLoader(study_dataset, batch_size = 5, shuffle = True)