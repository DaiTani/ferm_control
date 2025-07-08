# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import joblib
import argparse
import os
import time  # 关键修复
import gymnasium as gym  # 替换为Gymnasium
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from torch.utils.data import TensorDataset, DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 模型定义 -------------------------
class LSTMPredictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.pre_fc = torch.nn.Linear(input_dim, 16)
        self.lstm = torch.nn.LSTM(16, hidden_dim, n_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        x = torch.relu(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        return self.fc(out[:, -1, :]), hidden
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 强化学习环境 ----------------------
class FermentationEnv(gym.Env):
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
        
        # 状态空间（所有特征）
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(initial_state.shape[0],),
            dtype=np.float32
        )
        
        # 初始化状态
        self.initial_state = initial_state
        self.current_state = initial_state.copy()
        self.history_buffer = []
        self.hidden_state = model.init_hidden()

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        self.current_state = self.initial_state.copy()
        self.history_buffer = [self.current_state]
        self.hidden_state = self.model.init_hidden()
        return self.current_state, {}

    def step(self, action):
        """
        执行动作并返回新的状态、奖励、是否终止和附加信息
        """
        # 1. 应用动作（带边界约束）
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.current_state[1:11] = clipped_action  # 假设可调参数位于索引1-10
        
        # 2. 更新历史缓冲区
        self.history_buffer.append(self.current_state.copy())
        if len(self.history_buffer) > self.window_size:
            self.history_buffer.pop(0)
        
        # 3. 构建输入序列
        if len(self.history_buffer) < self.window_size:
            # 前填充初始状态
            padded = [self.initial_state]*(self.window_size - len(self.history_buffer)) + self.history_buffer
        else:
            padded = self.history_buffer[-self.window_size:]
        
        # 4. 数据预处理
        input_seq = np.array(padded)
        scaled_seq = self.scaler.transform(input_seq)
        tensor_seq = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
        
        # 5. 预测OD600并计算奖励
        with torch.no_grad():
            pred, self.hidden_state = self.model(tensor_seq, self.hidden_state)
        od_pred = pred.item()
        
        # 6. 设计奖励函数
        reward = self._calculate_reward(od_pred, action)
        
        # 7. 终止条件（模拟24小时）
        done = len(self.history_buffer) > 72  # 假设每小时一个时间步
        
        return self.current_state.copy(), reward, done, False, {}

    def _calculate_reward(self, od_pred, action):
        """复合奖励函数设计"""
        # 基础奖励：OD预测值
        base_reward = od_pred * 10  # 放大系数
        
        # 平滑性惩罚（动作变化率）
        if hasattr(self, 'last_action'):
            action_diff = np.linalg.norm(action - self.last_action)
            smooth_penalty = -0.5 * action_diff
        else:
            smooth_penalty = 0
        
        # 边界惩罚
        boundary_penalty = -10 * np.any([
            (action < self.action_space.low + 1e-3),
            (action > self.action_space.high - 1e-3)
        ])
        
        # 组合奖励
        total_reward = base_reward + smooth_penalty + boundary_penalty
        
        # 保存当前动作
        self.last_action = action.copy()
        
        return total_reward

# ---------------------- 数据预处理 ----------------------
def load_initial_state(dataset_path, timestamp, scaler):
    """加载初始状态并预处理"""
    try:
        df = pd.read_excel(os.path.join(dataset_path, "data.xlsx"))
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
        target_time = pd.to_datetime(timestamp, dayfirst=True)
        
        # 找到最接近的时间点
        closest_idx = (df['Timestamp'] - target_time).abs().idxmin()
        raw_features = df.iloc[closest_idx][[
            'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
            'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
            'dm_spump3', 'dm_spump4', 'induction'
        ]].values.astype(np.float32)
        
        # 处理累积量特征
        cumulative_indices = [4,5,6,7,8,9]
        for idx in cumulative_indices:
            if raw_features[idx] <= 0:
                raw_features[idx] = 0.001  # 防止零值
        
        # 归一化处理
        scaled = scaler.transform([raw_features])
        return scaled[0]
    
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" 智能发酵强化学习优化系统 v2.1")
    print("="*40)
    
    # 验证文件存在性
    required_files = {
        '模型权重': args.weights,
        '数据集': os.path.join(args.dataset, "data.xlsx"),
        '归一化器': os.path.join(args.dataset, "scaler_new_data.save")
    }
    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"[错误] {name}不存在: {path}")
            exit(1)

    # 加载归一化器
    try:
        scaler = joblib.load(required_files['归一化器'])
    except Exception as e:
        print(f"归一化器加载失败: {str(e)}")
        exit(1)

    # 初始化LSTM模型
    model = LSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['weights'])
        model.eval()
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)

    # 获取初始状态
    initial_state = load_initial_state(args.dataset, args.Timestamp, scaler)
    
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

    # 创建并行环境
    env = make_vec_env(
        lambda: FermentationEnv(model, scaler, initial_state, param_bounds),
        n_envs=4
    )

    # 配置PPO算法
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cpu"  # 强制使用CPU避免警告
    )

    # 训练回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./rl_models/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True
    )

    # 开始训练
    print("\n[状态] 开始强化学习训练...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=50000,
        callback=eval_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time

    # 获取最优参数
    optimal_params = []
    obs = env.reset()
    for _ in range(100):  # 100步稳定期
        action, _ = model_ppo.predict(obs, deterministic=True)
        optimal_params.append(action)
    final_params = np.mean(optimal_params[-20:], axis=0)

    # 结果输出
    print("\n" + "="*40)
    print(" 优化结果 ".center(40, '='))
    print(f" 训练耗时: {training_time:.1f}秒")
    print(f" 最终平均奖励: {np.mean([ep['r'] for ep in eval_callback.evaluations[-1][0]]):.2f}")
    print("-"*40)
    
    param_names = [
        "溶氧设定值(%)", "温度(℃)", "搅拌速率(rpm)",
        "O2流量(L/h)", "空气流量(L/h)", "补料泵1(mL/h)",
        "补料泵2(mL/h)", "补料泵3(mL/h)", "补料泵4(mL/h)", 
        "诱导剂浓度(mM)"
    ]
    for name, opt, init, bounds in zip(param_names, final_params, initial_state[1:11], param_bounds):
        safe_opt = np.clip(opt, bounds[0], bounds[1])
        print(f" {name:<15} {init:.2f} → {safe_opt:.2f} (范围: {bounds[0]:.1f}-{bounds[1]:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="发酵过程强化学习优化")
    parser.add_argument('--weights', type=str, required=True, help='LSTM模型权重路径')
    parser.add_argument('--dataset', type=str, required=True, help='数据集目录路径')
    parser.add_argument('--hidden_dim', type=int, default=32, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--Timestamp', type=str, required=True, help='优化起始时间戳(dd/mm/yyyy HH:MM)')
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
    except Exception as e:
        print(f"\n致命错误: {str(e)}")