# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import os
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from typing import List, Tuple

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: torch.nn.Module, scaler: object, initial_state: np.ndarray, param_bounds: List[Tuple[float, float]]):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.param_bounds = param_bounds
        self.window_size = 20
        
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in param_bounds], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(initial_state.shape[0],),
            dtype=np.float32
        )
        
        self.initial_state = initial_state.copy()
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        super().reset(seed=seed)
        self.current_state = self.initial_state.copy()
        self.history = [self.current_state.copy()]
        self.hidden = self.model.init_hidden()
        self.step_count = 0
        return self.current_state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.current_state[1:11] = np.clip(action, self.action_space.low, self.action_space.high)
        self.history.append(self.current_state.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        # 构建输入序列
        if len(self.history) < self.window_size:
            input_seq = [self.initial_state]*(self.window_size - len(self.history)) + self.history
        else:
            input_seq = self.history[-self.window_size:]
            
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)
        
        reward = pred.item() * 10
        done = self.step_count >= 72
        self.step_count += 1
        
        return self.current_state.copy(), reward, done, False, {}

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 兼容版LSTM模型 -------------------------
class CompatibleLSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 输入层（保持与旧版本兼容）
        self.pre_fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU()
        )
        
        # LSTM核心（保持原始参数结构）
        self.lstm = torch.nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        
        # 输出层（兼容新旧版本）
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.pre_fc(x)
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden
    
    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)
        
    def load_initial_state(self) -> np.ndarray:
        """安全加载初始状态"""
        try:
            df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
            target_time = pd.to_datetime(self.timestamp, dayfirst=True)
            
            closest_idx = (df['Timestamp'] - target_time).abs().idxmin()
            features = df.iloc[closest_idx][[
                'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
                'dm_spump3', 'dm_spump4', 'induction'
            ]].values.astype(np.float32)
            
            # 数据消毒
            features = np.nan_to_num(features, nan=0.001)
            features[4:10] = np.where(features[4:10] <= 0, 0.001, features[4:10])
            
            return self.scaler.transform([features])[0]
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            exit(1)

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" 智能发酵优化系统 v6.0（兼容版）")
    print("="*40)
    
    # 初始化数据
    data_loader = SafeDataLoader(args.dataset, args.Timestamp, os.path.join(args.dataset, "scaler_new_data.save"))
    initial_state = data_loader.load_initial_state()
    
    # 参数边界
    param_bounds = [
        (max(0.1, initial_state[1]-10), initial_state[1]+10),
        (initial_state[2]-1, initial_state[2]+1),
        (max(100, initial_state[3]-100), initial_state[3]+100),
        (max(10, initial_state[4]-50), initial_state[4]+50),
        (max(10, initial_state[5]-50), initial_state[5]+50),
        (max(0.1, initial_state[6]-0.5), initial_state[6]+0.5),
        (max(0.1, initial_state[7]-0.5), initial_state[7]+0.5),
        (max(0.1, initial_state[8]-0.5), initial_state[8]+0.5),
        (max(0.1, initial_state[9]-2), initial_state[9]+2),
        (0.001, 0.999)
    ]

    # 初始化模型
    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)
    
    # 加载权重（带兼容处理）
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        
        # 权重名称映射
        key_mapping = {
            'pre_fc.0.weight': 'pre_fc.0.weight',
            'pre_fc.0.bias': 'pre_fc.0.bias',
            'lstm.weight_ih_l0': 'lstm.weight_ih_l0',
            'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
            'lstm.bias_ih_l0': 'lstm.bias_ih_l0',
            'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
            'lstm.weight_ih_l1': 'lstm.weight_ih_l1',
            'lstm.weight_hh_l1': 'lstm.weight_hh_l1',
            'lstm.bias_ih_l1': 'lstm.bias_ih_l1',
            'lstm.bias_hh_l1': 'lstm.bias_hh_l1',
            'fc.0.weight': 'fc.0.weight',
            'fc.0.bias': 'fc.0.bias',
            'fc.2.weight': 'fc.2.weight',
            'fc.2.bias': 'fc.2.bias'
        }
        
        # 转换权重字典
        converted_weights = {}
        for old_key in checkpoint['weights']:
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                converted_weights[new_key] = checkpoint['weights'][old_key]
                print(f"映射权重: {old_key} → {new_key}")
            else:
                print(f"忽略不兼容参数: {old_key}")
        
        # 动态调整LSTM输入维度
        if converted_weights['lstm.weight_ih_l0'].shape[1] != model.lstm.weight_ih_l0.shape[1]:
            print("调整LSTM输入维度...")
            original_dim = model.lstm.weight_ih_l0.shape[1]
            saved_dim = converted_weights['lstm.weight_ih_l0'].shape[1]
            
            if saved_dim < original_dim:
                # 填充随机初始化
                pad = torch.zeros(64, original_dim - saved_dim, device=device)
                converted_weights['lstm.weight_ih_l0'] = torch.cat(
                    [converted_weights['lstm.weight_ih_l0'], pad], dim=1
                )
            else:
                # 截断多余维度
                converted_weights['lstm.weight_ih_l0'] = converted_weights['lstm.weight_ih_l0'][:, :original_dim]
        
        # 加载权重
        model.load_state_dict(converted_weights, strict=False)
        print("模型加载成功！")
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 创建环境
    env = make_vec_env(
        lambda: FermentationEnv(model, data_loader.scaler, initial_state, param_bounds),
        n_envs=4
    )

    # 配置PPO
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    # 训练
    print("\n[状态] 开始强化学习训练...")
    model_ppo.learn(
        total_timesteps=50000,
        callback=EvalCallback(env, best_model_save_path="./models/"),
        progress_bar=True
    )

    # 输出结果
    print("\n优化完成！最终参数建议：")
    obs = env.reset()
    action, _ = model_ppo.predict(obs, deterministic=True)
    for name, value, (low, high) in zip(
        ["溶氧", "温度", "搅拌", "O2流量", "空气流量", 
         "补料泵1", "补料泵2", "补料泵3", "补料泵4", "诱导剂"],
        action[0],
        param_bounds
    ):
        print(f"{name}: {np.clip(value, low, high):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能发酵优化系统")
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--Timestamp', type=str, required=True)
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n系统错误: {str(e)}")