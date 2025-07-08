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
from typing import List, Tuple, Dict
import warnings
from torch import nn

from rl_dataset import RealData

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------- 工具函数 ----------------------
def format_timestamp(minutes: float) -> str:
    hours = int(minutes // 60)
    remaining_minutes = int(minutes % 60)
    seconds = int((minutes - int(minutes)) * 60)
    return f"{hours}小时{remaining_minutes}分{seconds}秒"

# ---------------------- LSTM模型 ----------------------
class CompatibleLSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.act(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        out = self.relu(out)
        return out, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: nn.Module, scaler: object, dataset: pd.DataFrame, timestamp: float):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.window_size = 20
        self.dataset = dataset
        self.timestamp = timestamp

        # 参数的物理约束参考
        self.param_constraints = [
            (0, 0.0, None),  # 空气流量
            (1, 0.0, None),  # 溶氧
            (2, 3.0, 8.0),  # PH
            (3, 0.0, None),  # 搅拌
            (4, 0.0, None),  # 温度
            (5, 0.0, None),  # 氧气流量
            (6, 0.0, None),  # 补料泵1
            (7, 0.0, None),  # 补料泵2
            (8, 0.0, None),  # 补料泵3
            (9, 0.0, None),  # 补料泵4
            (10, 0.0, 1.0)  # 诱导剂浓度
        ]

        initial_state = self.reset()[0]
        initial_params = initial_state[:11].copy()
        epsilon = 1e-6
        # 参数的优化范围参考
        param_bounds = [
            (max(epsilon, initial_params[0] - 1.0), initial_params[0] + 1.0),  # 空气流量
            (max(epsilon, initial_params[1] - 1.0), initial_params[1] + 1.0),  # 溶氧
            (max(3.0, initial_params[2] - 0.5), min(8.0, initial_params[2] + 0.5)),  # PH
            (max(epsilon, initial_params[3] - 10.0), initial_params[3] + 10.0),  # 搅拌
            (max(0.0, initial_params[4] - 1.0), initial_params[4] + 1.0),  # 温度
            (max(epsilon, initial_params[5] - 1.0), initial_params[5] + 1.0),  # O2流量
            (max(epsilon, initial_params[6] - 1.0), initial_params[6] + 1.0),  # 补料泵1
            (max(epsilon, initial_params[7] - 1.0), initial_params[7] + 1.0),  # 补料泵2
            (max(epsilon, initial_params[8] - 1.0), initial_params[8] + 1.0),  # 补料泵3
            (max(epsilon, initial_params[9] - 0.5), initial_params[9] + 0.5),  # 补料泵4
            (epsilon, 1.0 - epsilon)  # 诱导剂浓度
        ]

        # 确保下限小于等于上限
        self.param_bounds = [(min(low, high), max(low, high)) for low, high in param_bounds]

        # 初始化动作和观测空间
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in self.param_bounds], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),  # 11个参数
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, **kwargs):
        # 加载当前时间戳之前的数据
        target_time = self.timestamp
        df_filtered = self.dataset[self.dataset['Timestamp'] <= target_time]

        # 使用过滤后的数据作为历史数据
        initial_state = df_filtered.iloc[-1][1:12].values.astype(np.float32)
        self.history = df_filtered.iloc[:, 1:12].values[-self.window_size:]
        self.hidden = self.model.init_hidden()
        self.step_count = 0

        # 返回观测值和空的信息字典
        return initial_state, {}

    def step(self, action: np.ndarray):
        # 应用物理约束
        for idx, lower, upper in self.param_constraints:
            if upper is not None:
                action[idx] = np.clip(action[idx], lower, upper)
            else:
                action[idx] = np.maximum(action[idx], lower)

        # 将动作剪切至优化范围
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.history = np.append(self.history[1:], [clipped_action], axis=0)

        # 构建输入序列
        input_seq = self.history[-self.window_size:]

        # 数据预处理
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

        # 使用模型预测
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)

        # 计算参数变化惩罚项
        prev_action = self.history[-2]
        param_change = np.linalg.norm(clipped_action - prev_action)
        penalty = 0.1 * param_change  # 惩罚系数可以调整

        # 改进后的奖励函数
        reward = pred.item() - penalty

        done = self.step_count >= 1
        self.step_count += 1

        return clipped_action, reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        preds = []
        current_state = self.history[-1].copy()
        hidden = self.model.init_hidden(batch_size=1)

        for _ in range(steps):
            # 预测下一步
            input_seq = np.array([current_state] * self.window_size)
            scaled_seq = self.scaler.transform(input_seq)
            tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

            with torch.no_grad():
                pred, hidden = self.model(tensor_input, hidden)

            preds.append(pred.item())
            current_state = np.roll(current_state, -1)
            current_state[-1] = pred.item()

        return preds


# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "=" * 40)
    print(" 生物过程优化系统 v8.5")
    print("=" * 40)

    # 数据加载
    data_loader = RealData(
        args.dataset,
        False,
    )

    # 加载LSTM模型
    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    model.load_state_dict(checkpoint['weights'], strict=True)
    print("✅ 模型加载成功")

    # 创建训练环境
    env = FermentationEnv(model, data_l, float(args.Timestamp))

    # 记录优化前的参数
    initial_obs, _ = env.reset()
    initial_params = initial_obs[:11]
    # 预测优化前的 OD600
    initial_od600_preds = env.predict_od600()

    # PPO算法配置，调整学习率和增加训练步数
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,  # 调整学习率
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    # 训练过程
    print("\n[状态] 开始强化学习训练...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=EvalCallback(env, best_model_save_path=f"./models_{int(time.time())}/", n_eval_episodes=3),
        progress_bar=True
    )
    print(f"训练完成，耗时 {time.time()-start_time:.1f}秒")

    # 打印训练结果
    obs, _ = env.reset()
    optimized_action, _ = model_ppo.predict(obs, deterministic=True)

    # 预测优化后的 OD600
    # 先将环境状态更新为优化后的参数
    env.history[-1] = optimized_action
    optimized_od600_preds = env.predict_od600()

    # 计算10步内优化前后的 OD600 预测值的平均值
    initial_od600_avg = np.mean(initial_od600_preds[:10])
    optimized_od600_avg = np.mean(optimized_od600_preds[:10])

    print(f"\n10步内优化前的 OD600 预测值的平均值：{initial_od600_avg:.4f}")
    print(f"10步内优化后的 OD600 预测值的平均值：{optimized_od600_avg:.4f}")

    # 如果10步内优化后的 OD600 预测值的平均值小于等于优化前的，则使用初始参数
    if optimized_od600_avg <= initial_od600_avg:
        optimized_action = initial_params

    # 参数对比显示
    param_names = [
        "空气流量(L/h)", "溶氧(%)", "PH", "搅拌(rpm)", "温度(℃)", "O2流量(L/h)",
        "补料泵1", "补料泵2", "补料泵3", "补料泵4", "诱导剂浓度"
    ]

    print("\n{:<15} {:<15} {:<15} {:<10}".format("参数", "原始值", "优化值", "变化率"))
    print("-" * 60)
    for i in range(11):
        orig = initial_params[i]
        optim = optimized_action[i]
        change = (optim - orig) / orig * 100 if orig != 0 else np.nan
        print(f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {change:+.1f}%" if not np.isnan(change) else
              f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {'N/A':<10}")

    print("\n优化前的 OD600 预测值（前 10 个时间步）：", initial_od600_preds[:10])
    print("优化后的 OD600 预测值（前 10 个时间步）：", optimized_od600_preds[:10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument('--weights', type=str, required=True, help="模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=2, help="LSTM层数")
    parser.add_argument('--Timestamp', type=str, required=True, help="初始时间戳（格式：小时:分钟）")

    args = parser.parse_args()
    main(args)