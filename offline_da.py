# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:15:32 2020

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

basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

data_dir = os.getcwd()+'\\data'
train_data = torchvision.datasets.MNIST(root=data_dir,
                                        train=True,
                                        download=True,
                                        transform=basic_transform)

test_data = torchvision.datasets.MNIST(root=data_dir,
                                        train=False,
                                        transform=basic_transform)

pretrain_loader = DataLoader(train_data, batch_size=60000, shuffle=True)
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

def augment_offline(data, label, augmentation_factor, augment):
  batch, width, height = data.shape
  data = data.cpu().numpy()
  label = label.cpu().numpy()
  augmented = []
  for i in range(augmentation_factor):
    augmented.append(augment(images=data))
  augmented = np.array(augmented).reshape(batch*augmentation_factor, width, height)
  new_label = np.tile(label, augmentation_factor)
  return augmented, new_label


def main(augment, epochs=1):
	#### 1. AUGMENT DATA BEFORE TRAINING ####
    enum_data = enumerate(pretrain_loader)
    _, (x_train, y_train) = next(enum_data)
    x_train = x_train.squeeze()

    start_augmentation = time.time()
    x_train_aug, y_train_aug = augment_offline(x_train, y_train, 8, augment)
    end_augmentation = time.time()
    augmentation_time = end_augmentation - start_augmentation
    print("augmentation time",augmentation_time)
    print("augmented training set shape",x_train_aug.shape)
    print("augmented label shape",y_train_aug.shape)

    trainingdata = MyData(x_train_aug, y_train_aug, return_perturb=False)
    trainingloader = DataLoader(trainingdata, batch_size=1024, shuffle=True)

    prior_model = ApproximateInvariance(input_dim=28*28, hidden_dim=512, output_dim=10, sample_size=1)
    # The prior isn't supposed to change during posterior training
    for param in prior_model.parameters():
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
          for batch_x, batch_y in trainingloader:
            optimizer.zero_grad()
            predictions = torch.zeros(30,len(batch_x),10)
            for j in range(30):
              predictions[j] = posterior_model(batch_x)
            pred = predictions.mean(0)
            kl = posterior_model.KL_loss()
            log_likelihood = crit(pred, batch_y)
            loss = kl + log_likelihood
            accuracy = compute_accuracy(pred, batch_y)

            accuracies_minibatch.append(accuracy)
            acc_epoch.append(accuracy)
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
    all_acc = evaluate_model(posterior_model, testloader, 10)
    acc_mean = round(all_acc.mean(),1)
    acc_std = round(all_acc.std(),1)
    print("time training (s):", int(time_training))
    print("Final test accuracy after 200 epoch: {} +/-{}".format(acc_mean, acc_std))


    result = {"augment":augment.name,
              "augmentation_time":augmentation_time,
              "time_training":time_training,
              "mean_test_accuracy":acc_mean,
              "std_test_accuracy":acc_std,
              "accuracies":accuracies_minibatch}


    df = pd.DataFrame(result, columns=["augment",
                                       "time_pretraining",
                                       "time_training",
                                       "mean_test_accuracy",
                                       "std_test_accuracy",
                                       "accuracies"])
    df.to_csv("results\\{}_offline_da.csv".format(augment.name))
    torch.save(posterior_model,"results\\offline_da_{}".format(augment.name))
