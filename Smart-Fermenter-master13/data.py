import subprocess
import os

# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))

# 定义Data5文件夹的路径
data5_folder = os.path.join(current_directory, "Data5")

# 定义要执行的脚本列表
scripts = [
    "data2.xlsx第一列转化1.py",
    #"data2.xlsx第一列转化2.py",
    "data2.xlsx第一列转化4.py"
]

# 依次执行每个脚本
for script in scripts:
    # 构建脚本的完整路径
    script_path = os.path.join(data5_folder, script)
    try:
        print(f"正在执行脚本: {script}")
        # 使用subprocess模块执行Python脚本
        result = subprocess.run(['python', script_path], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"脚本 {script} 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"脚本 {script} 执行失败，错误信息: {e.stderr}")