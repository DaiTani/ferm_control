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
from torch.utils.data import TensorDataset, DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 增强型LSTM模型 -------------------------
class EnhancedLSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.grad_norms = []  # 梯度监控
        
        # 网络结构
        self.pre_fc = torch.nn.Linear(input_dim, 16)
        self.lstm = torch.nn.LSTM(16, hidden_dim, n_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
        # 梯度裁剪
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -10, 10))
            
    def forward(self, x, hidden):
        x = torch.relu(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        output = self.fc(out[:, -1, :])
        
        # 梯度监控
        if output.requires_grad:
            output.register_hook(self._grad_hook)
            
        return output, hidden
    
    def _grad_hook(self, grad):
        norm = grad.norm().item()
        self.grad_norms.append(norm)
        if norm > 1e5:
            print(f"梯度异常: {norm:.2e}")
        return grad
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 诊断型强化学习环境 ----------------------
class DiagnosticFermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model, scaler, initial_state, param_bounds, window_size=20):
        super().__init__()
        
        # 环境参数
        self.model = model
        self.scaler = scaler
        self.window_size = window_size
        self.param_bounds = param_bounds
        
        # 动作空间（可调参数）
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in param_bounds], dtype=np.float32),
            dtype=np.float32
        )
        
        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(initial_state.shape[0],),
            dtype=np.float32
        )
        
        # 初始化
        self.initial_state = initial_state
        self._reset_internal_state()
        
    def _reset_internal_state(self):
        """内部状态重置"""
        self.current_state = self.initial_state.copy()
        self.history_buffer = [self.current_state]
        self.hidden_state = self.model.init_hidden()
        self.last_action = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_internal_state()
        return self.current_state, {}
    
    def step(self, action):
        # 1. 动作处理
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_state[1:11] = clipped_action
        
        # 2. 历史管理
        self.history_buffer.append(self.current_state.copy())
        if len(self.history_buffer) > self.window_size:
            self.history_buffer.pop(0)
            
        # 3. 序列构建
        padded = self._get_padded_sequence()
        
        # 4. 数据预处理
        input_seq = np.array(padded)
        scaled_seq = self.scaler.transform(input_seq)
        tensor_seq = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
        
        # 5. 预测与奖励
        with torch.no_grad():
            pred, self.hidden_state = self.model(tensor_seq, self.hidden_state)
        od_pred = pred.item()
        reward = self._calculate_reward(od_pred, clipped_action)
        
        # 6. 终止判断
        done = len(self.history_buffer) > 72  # 模拟24小时
        
        # 7. 调试输出
        if np.random.rand() < 0.01:  # 1%概率采样输出
            print(f"[DEBUG] Action: {clipped_action.round(2)} | Reward: {reward:.2f}")
            
        return self.current_state.copy(), reward, done, False, {}

    def _get_padded_sequence(self):
        """构建填充序列"""
        if len(self.history_buffer) < self.window_size:
            return [self.initial_state]*(self.window_size - len(self.history_buffer)) + self.history_buffer
        return self.history_buffer[-self.window_size:]

    def _calculate_reward(self, od_pred, action):
        """稳健奖励计算"""
        base_reward = od_pred * 10
        
        # 平滑性惩罚
        smooth_penalty = 0
        if self.last_action is not None:
            action_diff = np.linalg.norm(action - self.last_action)
            smooth_penalty = -0.5 * action_diff
            
        # 边界惩罚
        boundary_penalty = -10 * np.any([
            (action < self.action_space.low + 1e-3),
            (action > self.action_space.high - 1e-3)
        ])
        
        self.last_action = action.copy()
        return base_reward + smooth_penalty + boundary_penalty

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path, timestamp, scaler_path):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)
        
    def load_initial_state(self):
        """安全加载初始状态"""
        try:
            df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
            target_time = pd.to_datetime(self.timestamp, dayfirst=True)
            
            # 查找最近时间点
            closest_idx = (df['Timestamp'] - target_time).abs().idxmin()
            raw_features = df.iloc[closest_idx][[
                'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
                'dm_spump3', 'dm_spump4', 'induction'
            ]].values.astype(np.float32)
            
            # 数据清洗
            raw_features = self._sanitize_features(raw_features)
            return self.scaler.transform([raw_features])[0]
            
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            exit(1)
            
    def _sanitize_features(self, features):
        """数据消毒处理"""
        # 处理累积量
        cumulative_indices = [4,5,6,7,8,9]
        for idx in cumulative_indices:
            if features[idx] <= 0:
                features[idx] = 0.001
        
        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.001, posinf=1e5, neginf=-1e5)
        return features

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" 智能发酵优化系统 v3.0")
    print("="*40)
    
    # 初始化组件
    data_loader = SafeDataLoader(args.dataset, args.Timestamp, 
                               os.path.join(args.dataset, "scaler_new_data.save"))
    initial_state = data_loader.load_initial_state()
    
    # 构建LSTM模型
    model = EnhancedLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)
    
    # 加载预训练权重
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 定义参数边界
    param_bounds = [
        (max(0.1, initial_state[1]-10), initial_state[1]+10),      # 溶氧
        (initial_state[2]-1, initial_state[2]+1),                 # 温度
        (max(100, initial_state[3]-100), initial_state[3]+100),   # 搅拌
        (max(10, initial_state[4]-50), initial_state[4]+50),      # O2流量
        (max(10, initial_state[5]-50), initial_state[5]+50),      # 空气流量
        (max(0.1, initial_state[6]-0.5), initial_state[6]+0.5),   # 补料泵1
        (max(0.1, initial_state[7]-0.5), initial_state[7]+0.5),   # 补料泵2
        (max(0.1, initial_state[8]-0.5), initial_state[8]+0.5),   # 补料泵3
        (max(0.1, initial_state[9]-2), initial_state[9]+2),       # 补料泵4
        (0.001, 0.999)                                            # 诱导剂
    ]

    # 创建训练环境
    env = make_vec_env(
        lambda: DiagnosticFermentationEnv(model, data_loader.scaler, initial_state, param_bounds),
        n_envs=4
    )

    # 配置PPO算法
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
        gae_lambda=0.90,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu",
        tensorboard_log="./tensorboard_logs/"
    )

    # 训练监控
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./rl_models/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    # 分阶段训练
    print("\n[阶段1] 初始5000步验证...")
    model_ppo.learn(total_timesteps=3990, callback=eval_callback, progress_bar=True)
    
    print("\n[阶段2] 完整50000步训练...")
    model_ppo.learn(total_timesteps=45000, callback=eval_callback, reset_num_timesteps=False)

    # 结果分析
    print("\n梯度分析:")
    plt.plot(model.grad_norms)
    plt.yscale('log')
    plt.title("梯度变化趋势")
    plt.show()

    # 保存最终参数
    final_params = np.mean([model_ppo.predict(env.reset()[0])[0] for _ in range(100)], axis=0)
    print("\n优化参数:", final_params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能发酵强化学习优化系统")
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