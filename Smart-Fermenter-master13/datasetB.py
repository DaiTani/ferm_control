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
        self.timestamp = timestamp  # 支持时间戳筛选
        self.y_mean = y_mean
        self.y_std = y_std
        self.ws = ws  # 窗口大小
        self.stride = stride  # 步长
        print("Loading dataset...")  # 添加打印语句

        # 训练和测试的发酵批次
        self.train_fermentations = [8]  # 训练批次
        self.test_fermentations = [28]  # 测试批次

        # 累积变量和二进制变量
        self.cumulative_var = []  # 累积变量
        self.binary_var = ["induction"]  # 二进制变量

        # 输入和输出变量
        self.x_var = [
            "dm_air", "m_ls_opt_do", "m_ph", "m_stirrer", "m_temp", "dm_o2",
            "dm_spump1", "dm_spump2", "dm_spump3", "dm_spump4", "induction"
        ]
        self.y_var = y_var  # 输出变量

        # 使用指定批次计算归一化参数
        self.fermentation_norm_number = 22  # 归一化批次
        self.X_norm, self.Y_norm = self.load_data(fermentation_number=self.fermentation_norm_number)
        self.X_norm = self.cumulative2snapshot(self.X_norm[0])
        self.Y_norm = self.cumulative2snapshot(self.Y_norm[0])

        # 加载数据
        self.X, self.Y = self.load_data()

        # 预处理数据
        self.X = self.preprocess_data(self.X, norm_mode="z-score", ws=self.ws, stride=self.stride)
        self.Y = self.preprocess_labels(self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride)

        # 训练模式下打乱数据
        if self.train:
            np.random.seed(1234)
            idx = np.random.permutation(len(self.X))
            self.X = self.X[idx]
            self.Y = self.Y[idx]

    def load_data(self, fermentation_number=None):
        """加载数据"""
        X = []
        Y = []

        # 加载单个批次数据
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

        # 加载训练或测试批次数据
        fermentations = self.train_fermentations if self.train else self.test_fermentations
        for fn in fermentations:
            data = utils.load_data(
                work_dir=self.work_dir,
                fermentation_number=fn,
                data_file="data.xlsx",
                x_cols=self.x_var,
                y_cols=self.y_var,
            )
            X.append(data[0])
            Y.append(data[1])

        # 根据时间戳筛选数据
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
        """预处理数据"""
        mean, std = utils.get_norm_param(X=self.X_norm, x_cols=self.x_var)
        processed_X = []
        for x in X:
            x = self.cumulative2snapshot(x)
            x = self.normalise(x, mean, std, norm_mode)
            x = self.data2sequences(x, ws, stride)
            processed_X.append(x)
        return np.concatenate(processed_X, axis=0)

    def preprocess_labels(self, Y, norm_mode="z-score", ws=20, stride=10):
        """预处理标签"""
        mean, std = utils.get_norm_param(X=self.Y_norm, x_cols=self.y_var)
        self.y_mean = torch.tensor(mean[0])  # 假设单输出，取第一个元素
        self.y_std = torch.tensor(std[0])
        processed_Y = []
        for y in Y:
            y = self.cumulative2snapshot(y)
            y = self.normalise(y, mean, std, norm_mode)
            y = self.data2sequences(y, ws, stride)
            processed_Y.append(y)
        return np.concatenate(processed_Y, axis=0)

    def cumulative2snapshot(self, X):
        """将累积数据转换为快照数据"""
        X = np.copy(X)
        for cv in self.cumulative_var:
            if cv in self.x_var:
                idx = self.x_var.index(cv)
            elif cv in self.y_var:
                idx = self.y_var.index(cv)
            X[:, idx] = utils.cumulative2snapshot(X[:, idx])
        return X

    def normalise(self, X, mean, std, mode="z-score"):
        """归一化数据"""
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
        return utils.normalise(X, mean=mean, std=std, mode=mode, binary_var=binary_var_idx)

    def data2sequences(self, X, ws=20, stride=10):
        """将数据转换为序列"""
        return utils.data2sequences(X, ws, stride)

    def update_windows(self):
        """重新生成时间窗口"""
        self.X = self.preprocess_data(self.X, norm_mode="z-score", ws=self.ws, stride=self.stride)
        self.Y = self.preprocess_labels(self.Y, norm_mode="z-score", ws=self.ws, stride=self.stride)

    def __getitem__(self, index):
        """获取单个样本"""
        x = self.X[index]
        y = self.Y[index]
        return torch.tensor(x), torch.tensor(y)

    def __len__(self):
        """数据集大小"""
        return self.X.shape[0]

    def get_num_features(self):
        """获取特征数量"""
        return self.X.shape[-1]