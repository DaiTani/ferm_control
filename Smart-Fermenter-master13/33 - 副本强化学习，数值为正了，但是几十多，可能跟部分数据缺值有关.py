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
import warnings
from torch import nn

warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: nn.Module, scaler: object, initial_state: np.ndarray):
        super().__init__()
        self.model = model
        self.scaler = scaler
        self.window_size = 20
        
        initial_params = initial_state[1:11].copy()
        epsilon = 1e-6
        
        # 参数索引对应关系：
        # 0:溶氧 1:温度 2:搅拌 3:O2流量 4:空气流量
        # 5:补料泵1 6:补料泵2 7:补料泵3 8:补料泵4 9:诱导剂浓度
        self.param_bounds = [
            (max(epsilon, initial_params[0] - 10.0), initial_params[0] + 10.0),  # 溶氧
            (max(0.0, initial_params[1] - 1.0), initial_params[1] + 1.0),        # 温度
            (max(epsilon, initial_params[2] - 100.0), initial_params[2] + 100.0),# 搅拌
            (max(epsilon, initial_params[3] - 100.0), initial_params[3] + 100.0),# O2流量
            (max(epsilon, initial_params[4] - 100.0), initial_params[4] + 100.0),# 空气流量
            (max(epsilon, initial_params[5] - 1.0), initial_params[5] + 1.0),    # 补料泵1
            (max(epsilon, initial_params[6] - 1.0), initial_params[6] + 1.0),    # 补料泵2
            (max(epsilon, initial_params[7] - 1.0), initial_params[7] + 1.0),    # 补料泵3
            (max(epsilon, initial_params[8] - 5.0), initial_params[8] + 5.0),    # 补料泵4
            (epsilon, 1.0 - epsilon)                                             # 诱导剂浓度
        ]
        
        self.param_bounds = [(float(low), float(high)) for low, high in self.param_bounds]
        
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.param_bounds], dtype=np.float32),
            high=np.array([b[1] for b in self.param_bounds], dtype=np.float32),
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
        clipped_action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high
        )
        self.current_state[1:11] = clipped_action  # 正确修改位置1-10
        self.history.append(self.current_state.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)
            
        input_seq = self.history[-self.window_size:] if len(self.history) >= self.window_size \
            else [self.initial_state]*(self.window_size - len(self.history)) + self.history
            
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)
        
        self.current_state[0] = pred.item()
        reward = pred.item()
        done = self.step_count >= 72
        self.step_count += 1
        
        return self.current_state.copy(), reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        od_predictions = []
        current_state = self.current_state.copy()
        hidden = self.model.init_hidden()
        temp_history = self.history.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                input_seq = temp_history[-self.window_size:] if len(temp_history) >= self.window_size \
                    else [self.initial_state]*(self.window_size - len(temp_history)) + temp_history
                
                scaled_seq = self.scaler.transform(input_seq)
                tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
                
                pred, hidden = self.model(tensor_input, hidden)
                od_predictions.append(pred.item())
                
                new_state = current_state.copy()
                new_state[0] = pred.item()
                temp_history.append(new_state.copy())
                if len(temp_history) > self.window_size:
                    temp_history.pop(0)
        
        return od_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- LSTM模型 -------------------------
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

# ---------------------- 数据加载器 ----------------------
class SafeDataLoader:
    def __init__(self, dataset_path: str, timestamp: str, scaler_path: str):
        self.dataset_path = dataset_path
        self.timestamp = timestamp
        self.scaler = joblib.load(scaler_path)

    @staticmethod
    def time_to_minutes(time_str: str) -> float:
        """Convert HH:MM:SS to minutes"""
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
            raise ValueError(f"Invalid time format: {time_str}, required HH:MM:SS or MM:SS") from e
        
    def load_initial_state(self) -> np.ndarray:
        try:
            df = pd.read_excel(os.path.join(self.dataset_path, "datacs.xlsx"))
            
            # Convert timestamp column
            if df['Timestamp'].dtype == object:
                df['Timestamp'] = df['Timestamp'].astype(str).apply(self.time_to_minutes)
            
            # Validate target time
            target_time = float(self.timestamp)
            if target_time < 0:
                raise ValueError("Target timestamp cannot be negative")

            # Find closest record
            time_diff = (df['Timestamp'] - target_time).abs()
            if time_diff.min() > 60:  # Threshold: 1 hour
                print(f"Warning: Closest record is {time_diff.min():.2f} minutes away from target")
                
            closest_idx = time_diff.idxmin()
            
            # Extract features (11 features)
            features = df.iloc[closest_idx][[
                'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
                'dm_spump3', 'dm_spump4', 'induction'
            ]].values.astype(np.float32)
            
            # Data sanitization
            features = np.nan_to_num(features, nan=0.001)
            features[5:11] = np.where(features[5:11] <= 0, 0.001, features[5:11])
            
            scaled_features = self.scaler.transform([features])
            return scaled_features[0]
            
        except Exception as e:
            print(f"Data loading failed: {str(e)}")
            exit(1)

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" Intelligent Fermentation Optimization System v7.5")
    print("="*40)
    
    # Initialize components
    data_loader = SafeDataLoader(
        args.dataset, 
        args.Timestamp,
        os.path.join(args.dataset, "scaler_new_data.save")
    )
    
    initial_state = data_loader.load_initial_state()
    
    # Load prediction model
    model = CompatibleLSTMPredictor(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_layers=args.num_layers
    ).to(device)
    
    try:
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['weights'], strict=True)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        exit(1)

    # Create environment
    env = make_vec_env(
        lambda: FermentationEnv(model, data_loader.scaler, initial_state),
        n_envs=4
    )

    # Configure PPO
    model_ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=1,
        device="cpu"
    )

    # Training
    print("\n[Status] Starting RL training...")
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=10000,
        callback=EvalCallback(env, best_model_save_path="./models/", n_eval_episodes=3),
        progress_bar=True
    )
    print(f"Training completed in {time.time()-start_time:.1f}s")

    # Results
    print("\nOptimization results:")
    obs = env.reset()
    action, _ = model_ppo.predict(obs, deterministic=True)
    
    # Parameter comparison
    pred_env = FermentationEnv(model, data_loader.scaler, initial_state)
    
    def denormalize_params(params):
        dummy = np.zeros((1, 11))
        dummy[0, 1:11] = params  # 参数存储在位置1-10
        denorm = data_loader.scaler.inverse_transform(dummy)
        return denorm[0, 1:11]   # 返回10个参数 (索引0-9)

    original_params = denormalize_params(pred_env.initial_state[1:11])
    optimized_params = denormalize_params(
        np.clip(action[0], pred_env.action_space.low, pred_env.action_space.high)
    )

    # 物理约束修正（索引0-9对应10个参数）
    # 参数索引对应关系：
    # 0:溶氧 1:温度 2:搅拌 3:O2流量 4:空气流量 
    # 5:补料泵1 6:补料泵2 7:补料泵3 8:补料泵4 9:诱导剂浓度
    
    # 补料泵参数非负约束 (索引5-8)
    original_params[5:9] = np.clip(original_params[5:9], 0, None)
    optimized_params[5:9] = np.clip(optimized_params[5:9], 0, None)
    
    # 诱导剂浓度约束 (索引9)
    original_params[9] = np.clip(original_params[9], 0, 1)
    optimized_params[9] = np.clip(optimized_params[9], 0, 1)

    # Display results
    param_names = [
        "Dissolved Oxygen(%)",    # 0
        "Temperature(℃)",        # 1
        "Agitation(rpm)",        # 2
        "O2 Flow(L/h)",          # 3
        "Air Flow(L/h)",         # 4
        "Feed Pump1(mL/h)",      # 5
        "Feed Pump2(mL/h)",      # 6
        "Feed Pump3(mL/h)",      # 7
        "Feed Pump4(mL/h)",      # 8
        "Inducer Conc."          # 9
    ]

    print("\n{:<20} {:<15} {:<15} {:<10}".format("Parameter", "Original", "Optimized", "Change"))
    print("-" * 65)
    for i in range(10):  # 正确循环0-9
        orig = original_params[i]
        optim = optimized_params[i]
        
        if orig == 0:
            change_str = "N/A"
        else:
            change_rate = (optim - orig)/orig * 100
            change_str = f"{change_rate:+.1f}%"
        
        print(f"{param_names[i]:<20} {orig:<15.2f} {optim:<15.2f} {change_str:<10}")

    # OD prediction
    od_predictions = pred_env.predict_od600(steps=72)
    od_scaler = joblib.load(os.path.join(args.dataset, "scaler_new_data.save"))
    od_mean = od_scaler.mean_[0]
    od_std = od_scaler.scale_[0]
    od_predictions = [p * od_std + od_mean for p in od_predictions]
    
    print("\nOD600 Prediction:")
    print(f"Current: {od_predictions[0]:.3f}")
    print(f"12h: {od_predictions[12*1]:.3f}")  # 假设每10分钟一个数据点
    print(f"24h: {od_predictions[24*1]:.3f}")
    print(f"Peak: {max(od_predictions):.3f} at {od_predictions.index(max(od_predictions))//6}h")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fermentation Optimization System")
    parser.add_argument('--weights', type=str, required=True, help="Path to LSTM weights")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset directory")
    parser.add_argument('--hidden_dim', type=int, required=True, help="LSTM hidden dimension")
    parser.add_argument('--num_layers', type=int, required=True, help="LSTM layers")
    parser.add_argument('--Timestamp', type=str, required=True, 
                       help="Initial timestamp in minutes (e.g. 35.95 for 35m57s)")
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"❌ Runtime error: {str(e)}")