import os
import ast

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, LabelEncoder


from utils import normalize_fn
from methods.pca import PCA


class H36M_Dataset(Dataset):

    def __init__(self, path_to_data="", split="train", means=None, stds=None):
        self.split = split
        self.path_to_data = path_to_data
        self.load_data(normalize = True, remove_nonmoving_joints=True, means=means, stds=stds)
        self.num_classes = 4

    def load_data(self, normalize = True, remove_nonmoving_joints=True, means=None, stds=None):
        '''
        Load data, split into train and validation
        '''
        self.normalize = normalize
        self.remove_nonmoving_joints = remove_nonmoving_joints

        if self.split == "train":
            all_data = np.load(self.path_to_data+"/h36m_data/h36m_train_data.npy")
            labels = np.load(self.path_to_data+"/h36m_data/h36m_train_labels.npy")

        if self.split == "val":
            all_data = np.load(self.path_to_data+"/h36m_data/h36m_val_data.npy")
            labels = np.load(self.path_to_data+"/h36m_data/h36m_val_labels.npy")

        if self.split=="test" or self.split == "test1":
            all_data = np.load(self.path_to_data+"/h36m_data/h36m_test1_data.npy")
            labels = np.load(self.path_to_data+"/h36m_data/h36m_test1_labels.npy")

        # for deep learning (MS2)
        if self.split=="test2":
            all_data = np.load(self.path_to_data+"/h36m_data/h36m_test2_data.npy")
            labels = np.zeros([all_data.shape[0]])

        if self.remove_nonmoving_joints:
            nonmoving_joints = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
            moving_joints = np.setdiff1d(np.arange(32), nonmoving_joints)
            all_data = all_data[:, :, moving_joints, :]

        if self.normalize:
            if self.split == "train":
                self.means = all_data.mean(axis=0, keepdims=True)
                self.stds  = all_data.std(axis=0, keepdims=True)
            else:
                self.means = means
                self.stds = stds
            all_data = normalize_fn(all_data, self.means, self.stds)

        data = all_data[:, :50, :]
        regression_target = all_data[:, 50:, :]

        data = data.reshape([data.shape[0],  -1])
        regression_target = regression_target.reshape([regression_target.shape[0],  -1])

        self.data = data
        self.regression_target = regression_target
        self.labels = labels 

        self.regression_target_size = regression_target.shape[1]
        self.feature_dim = data.shape[1]

    def __getitem__(self, idx):
        return self.data[idx], self.regression_target[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0]

class FMA_Dataset(Dataset):
    def __init__(self, path_to_data="", split="train", means=None, stds=None):
        self.FMAPATH = 'fma_data/'
        self.split = split
        self.path_to_data = path_to_data
        self.data, self.regression_target, self.labels = self.load_data(normalize_inputs=True, normalize_outputs=True, means=means, stds=stds)
        self.feature_dim = self.data.shape[1]
        self.num_classes = 16
        self.regression_target_size = 1

    def load_data(self, normalize_inputs, normalize_outputs, means=None, stds=None):
        '''
        Load data, split into train and validation
        '''
        
        if self.split == "train":
            all_data = np.load(self.path_to_data+"/fma_data/fma_train_data.npy")
            all_labels = np.load(self.path_to_data+"/fma_data/fma_train_labels.npy")

        if self.split == "val":
            all_data = np.load(self.path_to_data+"/fma_data/fma_val_data.npy")
            all_labels = np.load(self.path_to_data+"/fma_data/fma_val_labels.npy")

        if self.split == "test" or self.split == "test1":
            all_data = np.load(self.path_to_data+"/fma_data/fma_test1_data.npy")
            all_labels = np.load(self.path_to_data+"/fma_data/fma_test1_labels.npy")
        
        # for deep learning (MS2)
        if self.split == "test2":
            all_data = np.load(self.path_to_data+"/fma_data/fma_test2_data.npy")
            all_labels = np.zeros([all_data.shape[0], 2])
        
        if normalize_inputs:
            if self.split == "train":
                self.means = all_data.mean(axis=0, keepdims=True)
                self.stds  = all_data.std(axis=0, keepdims=True)
            else:
                self.means = means
                self.stds = stds
            all_data = normalize_fn(all_data, self.means, self.stds)
            
        data = all_data
        regression_target = all_labels[...,0]
        labels = all_labels[...,1]
        
        if normalize_outputs:
            reg_means = regression_target.mean(axis=0, keepdims=True)
            reg_stds  = regression_target.std(axis=0, keepdims=True)
            regression_target = normalize_fn(regression_target, reg_means, reg_stds)
        
        return data.astype('float32'), regression_target.astype('float32'), labels.astype('int64')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.regression_target[idx], self.labels[idx]

class Movie_Dataset(Dataset):

    def __init__(self, path_to_data="", split="train", means=None, stds=None, task='regression'):
        self.split = split
        self.num_classes = 10
        self.path_to_data = path_to_data
        self.load_data(normalize = True, means=means, stds=stds)
        self.task = task
        self.num_classes = 10
        self.feature_dim = self.data.shape[1]
        self.regression_target_size = 1


    def load_data(self, normalize = True, means=None, stds=None):
        '''
        Load data, split into train and validation
        '''
        self.normalize = normalize

        if self.split == "train":
            data = np.loadtxt(self.path_to_data+"/Movie_data/movie_train_data.npy").astype(np.float32)
            labels = np.loadtxt(self.path_to_data+"/Movie_data/movie_train_labels.npy").astype(np.float32)

        if self.split == "val":
            data = np.loadtxt(self.path_to_data+"/Movie_data/movie_val_data.npy").astype(np.float32)
            labels = np.loadtxt(self.path_to_data+"/Movie_data/movie_val_labels.npy").astype(np.float32)

        if self.split == "test" or self.split=="test1":
            data = np.loadtxt(self.path_to_data+"/Movie_data/movie_test1_data.npy").astype(np.float32)
            labels = np.loadtxt(self.path_to_data+"/Movie_data/movie_test1_labels.npy").astype(np.float32)

        # for deep learning (MS2)
        if self.split=="test2":
            data = np.loadtxt(self.path_to_data+"/Movie_data/movie_test2_data.npy").astype(np.float32)
            labels = np.zeros([data.shape[0], 2])

        if self.normalize:
            if self.split == "train":
                self.means = data.mean(axis=0, keepdims=True)
                self.stds  = data.std(axis=0, keepdims=True)
            else:
                self.means = means
                self.stds = stds

            data = normalize_fn(data, self.means, self.stds)

        self.data = data
        self.labels = labels[...,1].astype(int)
        self.regression_target = labels[...,0] 

        self.regression_target = normalize_fn(self.regression_target, self.regression_target.mean(), self.regression_target.std())

    def __getitem__(self, idx):
        return self.data[idx], self.regression_target[idx], self.labels[idx]

    def __len__(self):
        return self.data.shape[0]    