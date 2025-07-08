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

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# ---------------------- 环境定义 ----------------------
class FermentationEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model: nn.Module, scaler: object, initial_state: np.ndarray, param_bounds: List[Tuple[float, float]]):
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
        input_seq = self.history[-self.window_size:] if len(self.history) >= self.window_size \
            else [self.initial_state]*(self.window_size - len(self.history)) + self.history
            
        scaled_seq = self.scaler.transform(input_seq)
        tensor_input = torch.FloatTensor(scaled_seq).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred, self.hidden = self.model(tensor_input, self.hidden)
        
        reward = pred.item() * 10
        done = self.step_count >= 72
        self.step_count += 1
        
        return self.current_state.copy(), reward, done, False, {}

    def predict_od600(self, steps: int = 72) -> List[float]:
        """预测指定步数的OD600值"""
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
                od_predictions.append(pred.item() * 10)
                
                new_state = current_state.copy()
                new_state[0] = pred.item() * 10
                temp_history.append(new_state.copy())
                if len(temp_history) > self.window_size:
                    temp_history.pop(0)
        
        return od_predictions

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- 兼容版LSTM模型 -------------------------
class CompatibleLSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # 与训练模型一致的预处理层
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        
        # LSTM核心（输入尺寸与hidden_dim一致）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        
        # 调整后的输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.act(self.pre_fc(x))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # 仅取最后一个时间步
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
        
    def load_initial_state(self) -> np.ndarray:
        """安全加载初始状态"""
        try:
            df = pd.read_excel(os.path.join(self.dataset_path, "data.xlsx"))
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True)
            
            target_time = pd.to_datetime(self.timestamp, dayfirst=True)
            closest_idx = (df['Timestamp'] - target_time).abs().idxmin()
            
            features = df.iloc[closest_idx][[
                'm_ph', 'm_ls_opt_do', 'm_temp', 'm_stirrer',
                'dm_o2', 'dm_air', 'dm_spump1', 'dm_spump2',
                'dm_spump3', 'dm_spump4', 'induction'
            ]].values.astype(np.float32)
            
            features = np.nan_to_num(features, nan=0.001)
            features[4:10] = np.where(features[4:10] <= 0, 0.001, features[4:10])
            
            return self.scaler.transform([features])[0]
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            print(f"请检查以下内容：")
            print(f"1. data.xlsx文件路径: {os.path.abspath(self.dataset_path)}")
            print(f"2. 时间戳格式是否正确: {self.timestamp} (示例: 01:01:58)")
            exit(1)

# ---------------------- 模型加载器 ----------------------
def load_compatible_weights(model: nn.Module, weight_path: str):
    """兼容新旧版本的权重加载函数"""
    try:
        checkpoint = torch.load(weight_path, map_location=device)
        
        # 检测权重格式
        if 'weights' in checkpoint:  # 旧版格式
            print("检测到旧版权重格式")
            weights = checkpoint['weights']
        else:  # 新版格式
            print("检测到新版权重格式")
            weights = checkpoint
            
        # 权重映射表（适配训练模型结构）
        key_mapping = {
            'pre_fc.weight': 'pre_fc.weight',
            'pre_fc.bias': 'pre_fc.bias',
            'lstm.weight_ih_l0': 'lstm.weight_ih_l0',
            'lstm.weight_hh_l0': 'lstm.weight_hh_l0',
            'lstm.bias_ih_l0': 'lstm.bias_ih_l0',
            'lstm.bias_hh_l0': 'lstm.bias_hh_l0',
            'lstm.weight_ih_l1': 'lstm.weight_ih_l1',
            'lstm.weight_hh_l1': 'lstm.weight_hh_l1',
            'lstm.bias_ih_l1': 'lstm.bias_ih_l1',
            'lstm.bias_hh_l1': 'lstm.bias_hh_l1',
            'fc.weight': 'fc.weight',
            'fc.bias': 'fc.bias'
        }
        
        converted_weights = {}
        for old_key in weights:
            if old_key in key_mapping:
                new_key = key_mapping[old_key]
                converted_weights[new_key] = weights[old_key]
                print(f"成功映射: {old_key} → {new_key}")
            else:
                print(f"忽略不兼容参数: {old_key}")
        
        # 加载权重
        model.load_state_dict(converted_weights, strict=True)
        print("模型加载成功！")
        return model
        
    except Exception as e:
        print(f"\n模型加载失败: {str(e)}")
        print("可能原因：")
        print("1. 模型结构不匹配（检查--hidden_dim和--num_layers参数）")
        print("2. 权重文件损坏（尝试重新训练模型）")
        print("3. 输入特征维度不一致（应为11个特征）")
        exit(1)

# ---------------------- 主程序 ----------------------
def main(args):
    print("\n" + "="*40)
    print(" 智能发酵优化系统 v6.3（修复兼容版）")
    print("="*40)
    
    # 初始化数据
    data_loader = SafeDataLoader(
        args.dataset, 
        args.Timestamp, 
        os.path.join(args.dataset, "scaler_new_data.save")
    )
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
    
    # 加载权重
    model = load_compatible_weights(model, args.weights)

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
    start_time = time.time()
    model_ppo.learn(
        total_timesteps=50000,
        callback=EvalCallback(env, best_model_save_path="./models/", n_eval_episodes=3),
        progress_bar=True
    )
    print(f"训练完成! 耗时: {time.time()-start_time:.2f}秒")

    # 输出结果
    print("\n优化完成！最终参数建议：")
    obs = env.reset()
    action, _ = model_ppo.predict(obs, deterministic=True)
    
    pred_env = FermentationEnv(model, data_loader.scaler, initial_state, param_bounds)
    pred_env.current_state[1:11] = np.clip(action[0], pred_env.action_space.low, pred_env.action_space.high)
    
    # 预测OD600
    od_predictions = pred_env.predict_od600(steps=72)
    
    print("\nOD600预测趋势：")
    print(f"当前值: {od_predictions[0]:.3f}")
    print(f"12小时预测: {od_predictions[12]:.3f}")
    print(f"24小时预测: {od_predictions[24]:.3f}")
    print(f"48小时预测: {od_predictions[48]:.3f}")
    print(f"峰值浓度: {max(od_predictions):.3f} (第{od_predictions.index(max(od_predictions))}小时)")
    
    print("\n优化操作参数：")
    param_names = [
        "溶氧(%)", "温度(℃)", "搅拌(rpm)", "O2流量(L/h)", 
        "空气流量(L/h)", "补料泵1(mL/h)", "补料泵2(mL/h)",
        "补料泵3(mL/h)", "补料泵4(mL/h)", "诱导剂浓度"
    ]
    for name, value, (low, high) in zip(param_names, action[0], param_bounds):
        final_value = np.clip(value, low, high)
        print(f"{name}: {final_value:.2f} (调整范围: {low:.1f}~{high:.1f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能发酵优化系统")
    parser.add_argument('--weights', type=str, required=True, help="预训练模型权重路径")
    parser.add_argument('--dataset', type=str, required=True, help="数据集目录路径")
    parser.add_argument('--hidden_dim', type=int, default=16, help="必须与训练时使用的--hidden_dim一致")
    parser.add_argument('--num_layers', type=int, default=2, help="必须与训练时使用的--num_layers一致")
    parser.add_argument('--Timestamp', type=str, required=True, help="初始时间戳(HH:MM:SS)")
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n系统错误: {str(e)}")
        print("常见问题排查:")
        print("1. 确认data.xlsx包含正确的特征列")
        print("2. 检查scaler_new_data.save文件是否存在")
        print("3. 验证时间戳格式示例: 01:01:58")
        print("4. 确认--hidden_dim和--num_layers参数与训练时完全一致")