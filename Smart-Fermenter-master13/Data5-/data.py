import subprocess

# 定义要执行的脚本列表
scripts = [
    "data2.xlsx第一列转化1.py",
    "data2.xlsx第一列转化2.py",
    "data2.xlsx第一列转化3.py"
]

# 依次执行每个脚本
for script in scripts:
    try:
        print(f"正在执行脚本: {script}")
        # 使用subprocess模块执行Python脚本
        result = subprocess.run(['python', script], check=True, text=True, capture_output=True)
        print(result.stdout)
        print(f"脚本 {script} 执行成功")
    except subprocess.CalledProcessError as e:
        print(f"脚本 {script} 执行失败，错误信息: {e.stderr}")