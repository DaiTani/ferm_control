from __future__ import print_function
import os
from os.path import join
import numpy as np
import sys
import utils
import torch
from torch.utils.data import Dataset
import glob
import pdb
import pandas as pd


class FermentationData(Dataset):
    def __init__(
        self, work_dir="./Data", train_mode=True, y_var=["od_600"], ws=20, stride=5, timestamp=None, y_mean=None, y_std=None
    ):
        self.work_dir = work_dir
        self.train = train_mode
        self.timestamp = timestamp  # 新增：支持时间戳筛选
        self.y_mean = y_mean
        self.y_std = y_std
        print("yyyy: ", y_mean)
        print("yyyy: ", y_std)	
        print("Loading dataset...")  # 添加打印语句
        # lists of number of fermentations for training and testing
        # Data
        self.train_fermentations = [
            8,  # 0.540000
            11,  # 3.670000
            12,  # 3.840000
            #### 14,  # 4.500000
            #16,  # 2.050000
            ### 17,  # 17.000000
            ### 19,  # 14.500000
            ### 20,  # 14.800000
            #22,  # 0.500000
            #23,  # 0.570000
            #24,  # 0.530000
            #25,  # 0.554000
            #26,  # 0.532000
            #27,  # 0.598000
            # 28,  # 0.674000
        ]
        self.test_fermentations = [28]

        # variables with cumulative values
        self.cumulative_var = [
            #"dm_o2",
            #"dm_air",
            #"dm_spump1",
            #"dm_spump2",
            #"dm_spump3",
            #"dm_spump4",
        ]
        # variables with binary values
        self.binary_var = ["induction"]

        # input variables
        self.x_var = [
            "dm_air",
            "m_ls_opt_do",
            "m_ph",
            "m_stirrer",
            "m_temp",
            "dm_o2",
            "dm_spump1",
            "dm_spump2",
            "dm_spump3",
            "dm_spump4",
            "induction",
        ]
        # output variable
        self.y_var = y_var

        # Using fermentation 16 for computing normalisation parameters
        self.fermentation_norm_number = 22  # 16
        self.X_norm, self.Y_norm = self.load_data(
            fermentation_number=self.fermentation_norm_number
        )
        self.X_norm = self.X_norm[0]
        self.X_norm = self.cumulative2snapshot(self.X_norm)
        self.Y_norm = self.Y_norm[0]
        self.Y_norm = self.cumulative2snapshot(self.Y_norm)  # 处理Y的累积数据

        # Loading data
        self.X, self.Y = self.load_data()

        # Initialize normalization parameters for labels
        #self.y_mean = None
        #self.y_std = None

        if self.train:
            self.ws, self.stride = (ws, stride)
        else:
            self.ws, self.stride = (ws, 1)  # Stride for test is set to 1

        # Preprocessing data
        self.X = self.preprocess_data(
            self.X, norm_mode="z-score", ws=self.ws, stride=self.stride
        )
        self.Y = self.preprocess_labels(
            self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride
        )

        # Shuffling for training
        if self.train:
            np.random.seed(1234)
            idx = np.random.permutation(len(self.X))

            self.X = self.X[idx]
            self.Y = self.Y[idx]

    def load_data(self, fermentation_number=None):
        # Load data for train/test fermentations
        X = []
        Y = []

        # Loading single fermentation data
        if fermentation_number is not None:
            data = utils.load_data(
                work_dir=self.work_dir,
                fermentation_number=fermentation_number,
                data_file="data.xlsx",
                x_cols=self.x_var,
                y_cols=self.y_var,
            )
            X.append(data[0])
            Y.append(data[1])

            return np.array(X), np.array(Y)

        # Loading train/test fermentations data
        if self.train:
            for fn in self.train_fermentations:
                data = utils.load_data(
                    work_dir=self.work_dir,
                    fermentation_number=fn,
                    data_file="data.xlsx",
                    x_cols=self.x_var,
                    y_cols=self.y_var,
                )
                X.append(data[0])
                Y.append(data[1])
        else:
            for fn in self.test_fermentations:
                data = utils.load_data(
                    work_dir=self.work_dir,
                    fermentation_number=fn,
                    data_file="data.xlsx",
                    x_cols=self.x_var,
                    y_cols=self.y_var,
                )
                X.append(data[0])
                Y.append(data[1])

        # 如果指定了时间戳，筛选对应数据
        if self.timestamp is not None:
            X_filtered = []
            Y_filtered = []
            for x, y in zip(X, Y):
                df = pd.DataFrame(x, columns=self.x_var)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
                df_filtered = df[df['Timestamp'] == self.timestamp]
                if not df_filtered.empty:
                    X_filtered.append(df_filtered[self.x_var].values)
                    Y_filtered.append(y[df.index.isin(df_filtered.index)])
            X = X_filtered
            Y = Y_filtered

        return np.array(X), np.array(Y)

    def preprocess_data(self, X, norm_mode="z-score", ws=20, stride=10):
        # Preprocess data
        mean, std = utils.get_norm_param(X=self.X_norm, x_cols=self.x_var)

        processed_X = []
        for i, x in enumerate(X):
            x = self.cumulative2snapshot(x)
            x = self.normalise(x, mean, std, norm_mode)
            x = self.data2sequences(x, ws, stride)

            processed_X.append(x)

        processed_X = np.concatenate(processed_X, axis=0)

        return processed_X

    def preprocess_labels(self, Y, norm_mode="z-score", ws=20, stride=10):
        # Preprocess labels
        mean, std = utils.get_norm_param(X=self.Y_norm, x_cols=self.y_var)  # 确保参数名称一致

        processed_Y = []
        for y in Y:
            y = self.cumulative2snapshot(y)  # 处理累积数据
            y = self.normalise(y, mean, std, norm_mode)  # 归一化
            y = self.data2sequences(y, ws, stride)  # 转换为序列

            processed_Y.append(y)

        return np.concatenate(processed_Y, axis=0)

    def cumulative2snapshot(self, X):
        # Trasform cumulative data to snapshot data
        X = np.copy(X)

        for cv in self.cumulative_var:
            if cv in self.x_var:
                idx = self.x_var.index(cv)
            elif cv in self.y_var:
                idx = self.y_var.index(cv)

            X[:, idx] = utils.cumulative2snapshot(X[:, idx])

        return X

    def normalise(self, X, mean, std, mode="z-score"):
        # Normalise data
        binary_var_idx = []
        for bv in self.binary_var:
            if bv in self.x_var:
                idx = self.x_var.index(bv)
                if idx < X.shape[1]:
                    binary_var_idx.append(idx)
            elif bv in self.y_var:
                idx = self.y_var.index(bv)
                if idx < X.shape[1]:
                    binary_var_idx.append(idx)

        return utils.normalise(
            X, mean=mean, std=std, mode=mode, binary_var=binary_var_idx
        )

    def data2sequences(self, X, ws=20, stride=10):
        # Transform data to sequences with default sliding window 20 and stride 10
        return utils.data2sequences(X, ws, stride)

    def polynomial_interpolation(self, data):
        # Compute polynomial interpolation
        data = data[:, 0]

        if np.isnan(data[0]):
            data[0] = 0

        return utils.polynomial_interpolation(data).reshape(-1, 1)

    def linear_local_interpolation(self, data):
        # Compute linear local interpolation
        data = data[:, 0]

        if np.isnan(data[0]):
            data[0] = 0

        return utils.linear_local_interpolation(data).reshape(-1, 1)

    def mix_interpolation(self, data):
        # Compute linear local interpolation
        data = data[:, 0]

        a, b = (0.5, 0.5)

        if np.isnan(data[0]):
            data[0] = 0

        return utils.mix_interpolation(data).reshape(-1, 1)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]

        x = torch.tensor(x)
        y = torch.tensor(y)

        return [x, y]

    def __len__(self):
        return self.X.shape[0]

    def get_num_features(self):
        return self.X.shape[-1]
		
    def update_windows(self):
        # 更新滑动窗口
        self.X = self.preprocess_data(
            self.X, norm_mode="z-score", ws=self.ws, stride=self.stride
        )
        self.Y = self.preprocess_labels(
            self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride
        )		