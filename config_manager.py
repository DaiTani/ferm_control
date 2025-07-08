"""
配置管理模块
保存、加载和管理配置
"""
import json
import os
import time

CONFIG_DIR = "configs"
LAST_OPERATION_FILE = os.path.join(CONFIG_DIR, "last_operation.json")

# 确保配置目录存在
os.makedirs(CONFIG_DIR, exist_ok=True)

def load_config(config_name):
    """加载指定名称的配置"""
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置 {config_name} 失败: {e}")
        return None

def save_config(config_name, config_data):
    """保存配置到文件"""
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存配置 {config_name} 失败: {e}")
        return False

def save_operation_config(config_name, config_data):
    """保存操作配置"""
    return save_config(config_name, config_data)

def list_configs():
    """列出所有可用的配置"""
    configs = []
    if os.path.exists(CONFIG_DIR):
        for file in os.listdir(CONFIG_DIR):
            if file.endswith(".json") and file != "last_operation.json":
                configs.append(file[:-5])
    return sorted(configs)

def get_last_operation():
    """获取最后一次操作的配置信息"""
    if not os.path.exists(LAST_OPERATION_FILE):
        return None
    try:
        with open(LAST_OPERATION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载最后一次操作信息失败: {e}")
        return None

def set_last_operation(config_name):
    """设置最后一次操作的配置"""
    try:
        last_op_data = {
            "config_name": config_name,
            "timestamp": time.time()
        }
        with open(LAST_OPERATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(last_op_data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"保存最后一次操作信息失败: {e}")
        return False

def delete_config(config_name):
    """删除指定配置"""
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    if os.path.exists(config_path):
        try:
            os.remove(config_path)
            return True
        except Exception as e:
            print(f"删除配置 {config_name} 失败: {e}")
    return False
