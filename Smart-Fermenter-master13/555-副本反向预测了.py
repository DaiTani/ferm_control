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

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.act(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device))

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: nn.Module, scaler: object, initial_state: np.ndarray):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.window_size = 20

        # 参数边界初始化
        initial_params = initial_state[:11].copy()
        epsilon = 1e-6
        self.param_bounds = [
            (max(3.0, initial_params[0] - 0.5), min(8.0, initial_params[0] + 0.5)),  # PH
            (max(epsilon, initial_params[1] - 1.0), initial_params[1] + 1.0),  # 溶氧
            (max(0.0, initial_params[2] - 1.0), initial_params[2] + 1.0),  # 温度
            (max(epsilon, initial_params[3] - 10.0), initial_params[3] + 10.0),  # 搅拌
            (max(epsilon, initial_params[4] - 1.0), initial_params[4] + 1.0),  # O2流量
            (max(epsilon, initial_params[5] - 1.0), initial_params[5] + 1.0),  # 空气流量
            (max(epsilon, initial_params[6] - 1.0), initial_params[6] + 1.0),  # 补料泵1
            (max(epsilon, initial_params[7] - 1.0), initial_params[7] + 1.0),  # 补料泵2
            (max(epsilon, initial_params[8] - 1.0), initial_params[8] + 1.0),  # 补料泵3
            (max(epsilon, initial_params[9] - 0.5), initial_params[9] + 0.5),  # 补料泵4
            (epsilon, 1.0 - epsilon)  # 诱导剂浓度
        ]

        # 动作空间定义
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in self.param_bounds], dtype=np.float32),
            dtype=np.float32
        )

        # 观测空间定义
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(initial_state.shape[0],),
            dtype=np.float32
        )

        self.initial_state = initial_state.copy()
        self.od_mean = self.scaler.mean_[0]
        self.od_std = self.scaler.scale_[0]
        self.reset()

    def reset(self, seed: int = None, options: dict = None) -> np.ndarray:
        super().reset(seed=seed)
        self.current_state = self.initial_state.copy()
        self.history = [self.scaler.inverse_transform(self.current_state.reshape(1, -1)).flatten()]
        self.hidden = self.model.init_hidden()
        self.step_count = 0
        return self.current_state.copy(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 动作裁剪
        clipped_action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        self.current_state[:11] = clipped_action
        self.history.append(self.scaler.inverse_transform(self.current_state.reshape(1, -1)).flatten().copy())

        # 维护滑动窗口
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 构建输入序列
        input_seq = self.history[-self.window_size:] if len(self.history) >= self.window_size \
            else [self.scaler.inverse_transform(self.initial_state.reshape(1, -1)).flatten()] * (self.window_size - len(self.history)) + self.history

        # 数据预处理
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

        # 模型预测
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)

        reward = pred.item()
        done = self.step_count >= 72
        self.step_count += 1

        return self.current_state.copy(), reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        """修正后的预测方法"""
        preds = []
        current_state = self.scaler.inverse_transform(self.current_state.reshape(1, -1)).flatten()
        temp_history = [current_state.copy()]

        # 获取标准化参数
        od_mean = self.od_mean
        od_std = self.od_std

        # 初始化隐藏状态
        hidden = self.model.init_hidden(batch_size=1)

        # 平滑参数
        smooth_window = 5
        smoothed_preds = []

        with torch.no_grad():
            for _ in range(steps):
                # 构建输入序列（原始数据）
                if len(temp_history) >= self.window_size:
                    input_seq = temp_history[-self.window_size:]
                else:
                    input_seq = [self.scaler.inverse_transform(self.initial_state.reshape(1, -1)).flatten()] * (self.window_size - len(temp_history)) + temp_history

                # 数据标准化
                scaled_seq = self.scaler.transform(input_seq)
                tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)

                # 模型预测
                pred, hidden = self.model(tensor_input, hidden)
                raw_pred = pred.item()

                # 逆标准化预测结果
                current_pred_denorm = raw_pred * od_std + od_mean

                # 滑动平均平滑
                smoothed_preds.append(current_pred_denorm)
                if len(smoothed_preds) > smooth_window:
                    smoothed_preds.pop(0)

                current_pred = np.mean(smoothed_preds) if len(smoothed_preds) >= smooth_window else current_pred_denorm

                # 更新状态（存储原始值）
                new_state = current_state.copy()
                new_state[0] = current_pred  # 更新 OD600 值
                temp_history.append(new_state.copy())
                if len(temp_history) > self.window_size:
                    temp_history.pop(0)

                # 更新 current_state 用于下一次预测
                current_state = new_state.copy()

                preds.append(current_pred)

        return preds

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)

    @staticmethod
    def time_to_minutes(time_str: str) -> float:
        try:
            parts = list(map(float, time_str.split(':')))
            if len(parts) == 3:
                hours, mins, secs = parts
            elif len(parts) == 2:
                hours, mins, secs = 0, parts[0], parts[1]
            else:
                raise ValueError
            return hours * 60 + mins + secs / 60
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid time format: {time_str}") from e

    def load_initial_state(self) -> Tuple[np.ndarray, float]:
        try:
            df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))

            required_columns = [
                'Timestamp', 'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2', 'dm_spump3',
                'dm_spump4', 'induction'
            ]

            # 数据验证
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

            # 时间处理
            if df['Timestamp'].dtype == object:
                df['Timestamp'] = df['Timestamp'].astype(str).apply(self.time_to_minutes)

            # 查找最近时间点
            target_time = float(self.timestamp)
            time_diff = (df['Timestamp'] - target_time).abs()
            closest_idx = time_diff.idxmin()

            # 特征处理
            features = df.iloc[closest_idx][required_columns[1:]].values.astype(np.float32)
            features = np.nan_to_num(features, nan=np.nanmean(features))

            # 数据标准化
            scaled_features = self.scaler.transform([features])
            return scaled_features[0], df.iloc[closest_idx]['Timestamp']

        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            exit(1)

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" 生物过程优化系统 v8.5")
    print("="*40)

    # 初始化数据加载
    data_loader = SafeDataLoader(
        args.dataset,
        args.Timestamp,
        os.path.join(args.dataset, "scaler_new_data.save")
    )

    # 加载初始状态
    initial_state, initial_timestamp = data_loader.load_initial_state()

    # 模型初始化
    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)

    try:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['weights'], strict=True)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        exit(1)

    # 创建训练环境
    env = make_vec_env(
        lambda: FermentationEnv(model, data_loader.scaler, initial_state),
        n_envs=4
    )

    # PPO算法配置
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
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

    # 结果展示
    print("\n优化结果:")
    obs = env.reset()
    action, _ = model_ppo.predict(obs, deterministic=True)

    # 参数逆标准化
    def denormalize_params(params):
        dummy = np.zeros((1, 11))
        dummy[0, :11] = params
        return data_loader.scaler.inverse_transform(dummy)[0, :11]

    # 参数后处理
    original_params = denormalize_params(initial_state[:11])
    optimized_params = denormalize_params(
        np.clip(action[0], env.envs[0].action_space.low, env.envs[0].action_space.high)
    )

    # 物理约束应用
    param_constraints = [
        (0, 3.0, 8.0),  # PH
        (1, 0.0, None),  # 溶氧
        (2, 0.0, None),  # 温度
        (5, 0.0, None),  # 空气流量
        (6, 0.0, None),  # 补料泵1
        (7, 0.0, None),  # 补料泵2
        (8, 0.0, None),  # 补料泵3
        (9, 0.0, None),  # 补料泵4
        (10, 0.0, 1.0)  # 诱导剂浓度
    ]

    for idx, min_val, max_val in param_constraints:
        original_params[idx] = np.clip(original_params[idx], min_val, max_val or np.inf)
        optimized_params[idx] = np.clip(optimized_params[idx], min_val, max_val or np.inf)

    # 参数对比显示
    param_names = [
        "PH", "溶氧(%)", "温度(℃)", "搅拌(rpm)", "O2流量(L/h)",
        "空气流量(L/h)", "补料泵1", "补料泵2", "补料泵3", "补料泵4", "诱导剂浓度"
    ]

    print("\n{:<15} {:<15} {:<15} {:<10}".format("参数", "原始值", "优化值", "变化率"))
    print("-" * 60)
    for i in range(11):
        orig = original_params[i]
        optim = optimized_params[i]
        change = (optim - orig)/orig * 100 if orig != 0 else np.nan
        print(f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {change:+.1f}%" if not np.isnan(change) else
              f"{param_names[i]:<15} {orig:<15.2f} {optim:<15.2f} {'N/A':<10}")

    # OD600预测展示
    pred_env = FermentationEnv(model, data_loader.scaler, initial_state)
    od_predictions = pred_env.predict_od600(steps=72)

    # 创建新的环境实例，使用原始参数进行OD600预测
    original_pred_env = FermentationEnv(model, data_loader.scaler, initial_state)
    original_od_predictions = original_pred_env.predict_od600(steps=72)

    print("\nOD600预测趋势:")
    print("优化后参数:")
    print(f"初始值: {od_predictions[0]:.3f}")
    print(f"6小时后: {od_predictions[24]:.3f}")
    print(f"12小时后: {od_predictions[48]:.3f}")
    print(f"峰值浓度: {max(od_predictions):.3f}")

    print("\n原始参数:")
    print(f"初始值: {original_od_predictions[0]:.3f}")
    print(f"6小时后: {original_od_predictions[24]:.3f}")
    print(f"12小时后: {original_od_predictions[48]:.3f}")
    print(f"峰值浓度: {max(original_od_predictions):.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生物过程优化系统")
    parser.add_argument('--weights', type=str, required=True, help="模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=64, help="LSTM隐藏层维度")
    parser.add_argument('--num_layers', type=int, default=2, help="LSTM层数")
    parser.add_argument('--Timestamp', type=str, required=True, 
                       help="初始时间戳（格式：小时:分钟，例如 35:57 对应35分57秒）")
    
    args = parser.parse_args()
    main(args)