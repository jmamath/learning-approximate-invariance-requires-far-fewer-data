# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
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
#from keras.datasets import mnist
from imgaug import augmenters as iaa
#from keras.utils import np_utils
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#@title Early Stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):

        score = -val_loss
        # 1st iteration
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # torch.save(model.state_dict(), path+'/checkpoint.pt')
        self.val_loss_min = val_loss

# @title Dataset class
class MyData(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels, return_perturb=False, sample_size=None, augmentation=None, training=False):
        'Initialization'
        self.labels = labels
        self.data = data
        self.return_perturb = return_perturb
        self.augmentation = augmentation
        self.sample_size = sample_size
        self.training = training

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.data[index]
        h, w = X.shape
        # Load data and get label
        y = self.labels[index]
        if self.return_perturb==False:
          X = X.reshape(-1)
          return X,y
        elif self.sample_size > 1:
          X = X.cpu()
          y = y.cpu()
          X_repeated = np.tile(X, [self.sample_size, 1, 1]) # Because we want X.shape = (sample_size, 28,28)
          y_repeated = np.tile(y, [self.sample_size, 1])  # Because we want y.shape = (sample_size, 10)
          X_aug = self.augmentation(images=X_repeated)
          if self.training:
            # import pdb; pdb.set_trace()
            X_repeated = X_repeated.reshape(self.sample_size,-1)
            X_aug = X_aug.reshape(self.sample_size,-1)
          return X_repeated, X_aug, y_repeated
        else:
          X_aug = self.augmentation(images=X)
          X_aug = X_aug.reshape(-1)
          X = X.reshape(-1)
          return X, X_aug, y

# @title Gaussian Layer class
class GaussianLayer(nn.Module):
    def __init__(self, shape, standard=False):
        super(GaussianLayer, self).__init__()
        self.shape = shape
        if standard is True:
          self.mu = nn.Parameter(torch.zeros(shape))
          self.log_var = nn.Parameter(torch.zeros(shape))
        else:
          self.mu = nn.Parameter(torch.rand(shape))
          self.log_var = nn.Parameter(torch.rand(shape))

    def forward(self, num_samples=1):
        if not isinstance(num_samples, tuple):
            num_samples = (num_samples,)
        eps_shape = num_samples + self.shape
        eps = torch.randn(eps_shape) # ~ N(0,I)
        return self.mu + torch.exp(self.log_var) * eps

    def entropy(self):
        distribution = Normal(loc=self.mu, scale=self.log_var.exp())
        return distribution.entropy().mean()


# @title Invariant Prior class
############### 2. CREATE THE MODEL ###############
class ApproximateInvariance(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sample_size, prior=GaussianLayer):
        super(ApproximateInvariance, self).__init__()
        self.prior = prior
        self.sample_size = sample_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_1 = prior((hidden_dim, input_dim), standard=True)
        self.bias_1 = prior((hidden_dim,), standard=True)
        self.weight_2 = prior((output_dim, hidden_dim), standard=True)
        self.bias_2 = prior((output_dim,), standard=True)

    def batch_forward(self, x, x_aug):
        # We remove the num_sample dimension if it is equal to one.
        w1 = self.weight_1().squeeze(0)
        b1 = self.bias_1()
        w2 = self.weight_2().squeeze(0)
        b2 = self.bias_2()

        x = F.linear(x, w1, b1)
        x = F.relu(x)
        x = F.linear(x, w2, b2)
        x = F.softmax(x) + 1e-8

        x_aug = F.linear(x_aug, w1, b1)
        x_aug = F.relu(x_aug)
        x_aug = F.linear(x_aug, w2, b2)
        x_aug = F.softmax(x_aug) + 1e-8
        return x, x_aug

    def forward(self, x, x_aug):
        """
        We need to compute the output of the neural network for the input x
        and the augmented input x_aug with the same weights. And we need
        to sample a new set of weights for each augmentation, hence the loop
        Input:
          x: torch Tensor. shape = (batch_size, num_sample, input_dim)
          x_aug: has the same attribute as x. but here for each num_sample there is a different augmentation
          while for x the tensor is repeated to leverage broadcasting.
        """
        if self.sample_size > 1:
          batch_size, num_samples, _ = x.shape
          results = torch.zeros(batch_size, num_samples, self.output_dim)
          results_aug = torch.zeros_like(results)
          for i in range(num_samples):
            results[:,i], results_aug[:,i] = self.batch_forward(x[:,i], x_aug[:,i])
        else:
          results, results_aug = self.batch_forward(x, x_aug)
        return results, results_aug

    def entropy(self):
        """
        Each weight computes its own entropy
        """
        entropy_w1 = self.weight_1.entropy()
        entropy_b1 = self.bias_1.entropy()
        entropy_w2 = self.weight_2.entropy()
        entropy_b2 = self.bias_2.entropy()
        return entropy_w1 + entropy_b1 + entropy_w2 + entropy_b2

def kl_div_output(pred1, pred2, sample_size):
    """
    This function computes the KL divergence between the output of
    the standard neural network and and neural network with augmented data
    Input:
        pred1. Float tensor. K-class softmax prediction of network 1
        pred2. Float tensor. K-class softmax prediction of network 2
    Output:
        kl_div. Float. The KL divergence between the two
    """
    if sample_size > 1:
      batch_size, num_sample, output_dim = pred1.shape
      log_ratio = torch.log(pred1/pred2)
      kl_div = torch.mean(pred1 * log_ratio, axis=[0,1]) # Average over num_sample and batches
      return kl_div.sum()
    else:
      log_ratio = torch.log(pred1/pred2)
      kl_div = torch.mean(pred1 * log_ratio, axis=0) # Average over batches
      return kl_div.sum()



# @title Bayes by Backprogagation class
class BayesbyBackprop(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, prior):
        super(BayesbyBackprop, self).__init__()
        self.prior = prior
        self.weight_1 = GaussianLayer((hidden_dim, input_dim))
        self.bias_1 = GaussianLayer((hidden_dim,))
        self.weight_2 = GaussianLayer((output_dim, hidden_dim))
        self.bias_2 = GaussianLayer((output_dim,))

    def forward(self, x):
        # We remove the num_sample dimension if it is equal to one.
        w1 = self.weight_1().squeeze(0)
        b1 = self.bias_1()
        w2 = self.weight_2().squeeze(0)
        b2 = self.bias_2()

#        import pdb; pdb.set_trace()

        x = F.linear(x, w1, b1)
        x = F.selu(x)
        x = F.linear(x, w2, b2)
        x = F.selu(x)

        return x

    def sample(self, num_samples=5):
        w1_samples = self.weight_1(num_samples=num_samples).view((num_samples, -1))
        b1_samples = self.bias_1(num_samples=num_samples).view((num_samples, -1))
        w2_samples = self.weight_2(num_samples=num_samples).view((num_samples, -1))
        b2_samples = self.bias_2(num_samples=num_samples).view((num_samples, -1))

        gen_weights = torch.cat([w1_samples, b1_samples, w2_samples, b2_samples], 1)

        return gen_weights

    def __kl(self, mu_1, log_var_1, mu_2, log_var_2):
        """
        KL divergence between two univariate Gaussian
        """
        var_1 = log_var_1.exp()
        var_2 = log_var_2.exp()
        kl = torch.mean(log_var_2-log_var_1 + (var_1.pow(2)-var_2.pow(2) + (mu_1-mu_2).pow(2))/(2 * var_2.pow(2)))
        return kl

    def KL_loss(self):
        kl_w1 = self.__kl(self.weight_1.mu, self.weight_1.log_var, self.prior.weight_1.mu, self.prior.weight_1.log_var)
        kl_b1 = self.__kl(self.bias_1.mu, self.bias_1.log_var, self.prior.bias_1.mu, self.prior.bias_1.log_var)
        kl_w2 = self.__kl(self.weight_2.mu, self.weight_2.log_var, self.prior.weight_2.mu, self.prior.weight_2.log_var)
        kl_b2 = self.__kl(self.bias_2.mu, self.bias_2.log_var, self.prior.bias_2.mu, self.prior.bias_2.log_var)
        return (kl_w1 + kl_w2 + kl_b1 + kl_b2)/4
