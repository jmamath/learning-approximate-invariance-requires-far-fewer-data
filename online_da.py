# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:51:25 2020

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import os
import time

import torch
from torch.distributions import Normal
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import seaborn as sns
import imgaug
from imgaug import augmenters as iaa
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from classes import *

class AddUniformNoise(object):
    def __init__(self):
      super(AddUniformNoise, self).__init__()

    def __call__(self, tensor):
        return tensor + torch.rand(tensor.size()).cpu()

basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

rotation = transforms.Compose([transforms.RandomRotation(20, fill=(0,)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])

rot_noise = transforms.Compose([transforms.RandomRotation(20, fill=(0,)),
                            transforms.ToTensor(),
                            AddUniformNoise(),
                            transforms.Normalize((0.1307,), (0.3081,))])

noise =  transforms.Compose([transforms.ToTensor(),
                            AddUniformNoise(),
                            transforms.Normalize((0.1307,), (0.3081,))])

data_dir = os.getcwd()+'\\data'

test_data = torchvision.datasets.MNIST(root=data_dir,
                                        train=False,
                                        transform=basic_transform)

testloader = DataLoader(test_data, batch_size=1024, shuffle=True)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device
device = get_device()

def compute_accuracy(pred, y):
  _, predicted = torch.max(F.softmax(pred), 1)
  total = len(pred)
  correct = (predicted == y).sum()
  accuracy = 100 * correct.cpu().numpy() / total
  return accuracy

def evaluate_model(model, loader, repetition):
  all_accuracies = np.zeros(repetition)
  for i in range(repetition):
    acc_final = []
    for x, y in loader: # batch_level
      # import pdb; pdb.set_trace()
      x = x.to(device).squeeze().reshape(len(x),-1)
      y = y.to(device)

      predictions = torch.zeros(100,len(x),10)
      for j in range(100):
        predictions[j] = model(x)
      pred = predictions.mean(0)
      accuracy = compute_accuracy(pred, y)
      acc_final.append(accuracy)
    all_accuracies[i] = np.array(acc_final).mean()
  return all_accuracies

def main(augment_name, epochs=1):
    if augment_name=="rotation":
        augment=rotation
    elif augment_name=="noise":
        augment=noise
    else:
        augment=rot_noise

    train_data = torchvision.datasets.MNIST(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=rotation)

    trainloader = DataLoader(train_data, batch_size=1024, shuffle=True)

    prior_model = ApproximateInvariance(input_dim=28*28, hidden_dim=512, output_dim=10, sample_size=1)
    for param in prior_model.parameters():
        #freeze the prior
        param.requires_grad = False
    posterior_model = BayesbyBackprop(input_dim=28*28, hidden_dim=512, output_dim=10, prior=prior_model)
    optimizer = torch.optim.Adam(posterior_model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()

    likelihood_losses = []
    kl_losses = []
    accuracies = []
    accuracies_minibatch = []

    # TRAINING LOOP
    start = time.time()
    with trange(epochs) as pbar:
        for i in pbar:
          acc_epoch = []
          loglikelihood_epoch = []
          kl_epoch = []
          for batch_x, batch_y in trainloader:
            batch_x = batch_x.to(device).squeeze().reshape(len(batch_x),-1)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predictions = torch.zeros(30,len(batch_x),10)
            for j in range(30):
              predictions[j] = posterior_model(batch_x)
            pred = predictions.mean(0)
            kl = posterior_model.KL_loss()
            log_likelihood = crit(pred, batch_y)
            loss = kl + log_likelihood
            accuracy = compute_accuracy(pred, batch_y)

            acc_epoch.append(accuracy)
            accuracies_minibatch.append(accuracy)
            kl_epoch.append(kl.cpu().detach().numpy())
            loglikelihood_epoch.append(log_likelihood.cpu().detach().numpy())

            loss.backward()
            optimizer.step()
            # import pdb; pdb.set_trace()
            # load the last checkpoint with the best model
            # pre_train_model.load_state_dict(torch.load('checkpoint.pt'))
          likelihood_losses.append(np.array(loglikelihood_epoch).mean())
          kl_losses.append(np.array(kl_epoch).mean())
          accuracies.append(np.array(acc_epoch).mean())
          pbar.set_postfix(kl=kl_losses[i],  log_likelihood=likelihood_losses[i], accuracy=accuracies[i])
    end = time.time()
    time_training = end - start
    print("time (s): {}".format(time_training))

    # TEST TIME
    all_acc = evaluate_model(posterior_model, testloader, 10)
    acc_mean = round(all_acc.mean(),1)
    acc_std = round(all_acc.std(),1)
    print("time training (s):", int(time_training))
    print("Final test accuracy after {} epoch: {} +/-{}".format(epochs, acc_mean, acc_std))

    result = {"augment":augment_name,
          "time_training":time_training,
          "mean_test_accuracy":acc_mean,
          "std_test_accuracy":acc_std,
          "accuracies":accuracies_minibatch}


    df = pd.DataFrame(result, columns=["augment",
                                       "time_training",
                                       "mean_test_accuracy",
                                       "std_test_accuracy",
                                       "accuracies"])
    df.to_csv("results\\{}_online_da.csv".format(augment_name))
    torch.save(posterior_model,"results\\online_da_{}".format(augment_name))
