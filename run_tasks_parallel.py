import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import json
import psutil

# 确保输出目录存在
os.makedirs("output_backup", exist_ok=True)

# 定义任务列表
tasks = [
    # "--tasks deepseek-v1_vs_doubao",
    # "--tasks deepseek-v1_vs_gpt_4o",
    # "--tasks doubao_vs_gpt_4o",
    # "--tasks doubao_vs_deepseek-v1",
    # "--tasks gpt_4o_vs_deepseek-v1",
    "--tasks gpt_4o_vs_doubao",
    # "--tasks gpt_4o_vs_deepseek-v1",
]

# 保存进程信息的文件
PROCESS_INFO_FILE = "running_tasks.json"

def save_process_info(pid, task_name, output_file):
    """保存进程信息到文件"""
    process_info = {
        "pid": pid,
        "task_name": task_name,
        "output_file": output_file,
        "start_time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # 读取现有信息
    if os.path.exists(PROCESS_INFO_FILE):
        with open(PROCESS_INFO_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_info = json.load(f)
            except json.JSONDecodeError:
                existing_info = []
    else:
        existing_info = []
    
    # 添加新信息
    existing_info.append(process_info)
    
    # 保存更新后的信息
    with open(PROCESS_INFO_FILE, 'w', encoding='utf-8') as f:
        json.dump(existing_info, f, indent=4, ensure_ascii=False)

# 定义运行任务的函数
def run_task(task):
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 从task中提取任务名称
    task_name = task.split(" ")[-1]
    # 构建输出文件名
    output_file = f"output_backup/{task_name}_{current_time}.txt"
    
    # 修改为正确的评估脚本路径
    command = f"C:/anaconda/envs/dsgbench/python.exe multiprocess_eval_tasks.py {task}"
    
    # 使用重定向将输出写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(command, shell=True, stdout=f, stderr=f)
        pid = process.pid
        print(f"Started task '{task_name}' with PID: {pid}")
        save_process_info(pid, task_name, output_file)
        return process

# 使用线程池并行运行任务
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        print("Starting parallel execution of tasks...")
        processes = list(executor.map(run_task, tasks))
        print("\nAll tasks started. Process information saved to", PROCESS_INFO_FILE)