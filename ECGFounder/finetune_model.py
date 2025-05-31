import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from net1d import Net1D

import torch.nn as nn
import torch


def ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=False):
  model = Net1D(
      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  # Load checkpoint with safer loading
  checkpoint = torch.load(pth, map_location=device, weights_only=False)
  state_dict = checkpoint['state_dict']

  # Remove classifier weights from pre-trained model
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  # Check for NaN/Inf in loaded weights
  for name, param in state_dict.items():
      if torch.isnan(param).any() or torch.isinf(param).any():
          print(f"Warning: Found NaN/Inf in parameter {name}, initializing randomly")
          if len(param.shape) >= 2:
              torch.nn.init.xavier_uniform_(param)
          else:
              torch.nn.init.zeros_(param)

  # Load state dict
  missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
  if missing_keys:
      print(f"Missing keys in model: {missing_keys}")
  if unexpected_keys:
      print(f"Unexpected keys in checkpoint: {unexpected_keys}")

  # Initialize new classifier layer properly
  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)
  
  # Initialize classifier weights properly to prevent NaN
  torch.nn.init.xavier_uniform_(model.dense.weight)
  torch.nn.init.zeros_(model.dense.bias)
  
  # freezing model
  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
            param.requires_grad = False

  model.to(device)

  return model


def ft_1lead_ECGFounder(device, pth, n_classes,linear_prob=False):
  model = Net1D(
      in_channels=1, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=False, 
      use_bn=False,
      use_do=False,
      n_classes=n_classes)

  # Load checkpoint with safer loading
  checkpoint = torch.load(pth, map_location=device, weights_only=False)
  state_dict = checkpoint['state_dict']

  # Remove classifier weights from pre-trained model
  state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')} 

  # Check for NaN/Inf in loaded weights
  for name, param in state_dict.items():
      if torch.isnan(param).any() or torch.isinf(param).any():
          print(f"Warning: Found NaN/Inf in parameter {name}, initializing randomly")
          if len(param.shape) >= 2:
              torch.nn.init.xavier_uniform_(param)
          else:
              torch.nn.init.zeros_(param)

  # Load state dict
  missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
  if missing_keys:
      print(f"Missing keys in model: {missing_keys}")
  if unexpected_keys:
      print(f"Unexpected keys in checkpoint: {unexpected_keys}")

  # Initialize new classifier layer properly
  model.dense = nn.Linear(model.dense.in_features, n_classes).to(device)
  
  # Initialize classifier weights properly to prevent NaN
  torch.nn.init.xavier_uniform_(model.dense.weight)
  torch.nn.init.zeros_(model.dense.bias)

  if linear_prob == True: 
    for name, param in model.named_parameters():
        if 'dense' not in name:  # no freezing last layer
            param.requires_grad = False
  model.to(device)

  return model

