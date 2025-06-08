import json
import os
import psutil
import argparse
from datetime import datetime

PROCESS_INFO_FILE = "running_tasks.json"

def load_process_info():
    """加载进程信息"""
    if not os.path.exists(PROCESS_INFO_FILE):
        print("No running tasks information found.")
        return []
    
    with open(PROCESS_INFO_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("Error reading process information file.")
            return []

def kill_process(pid):
    """终止指定PID的进程"""
    try:
        process = psutil.Process(pid)
        process.terminate()
        process.wait(timeout=3)  # 等待进程终止
        return True
    except psutil.NoSuchProcess:
        print(f"Process {pid} does not exist.")
        return False
    except psutil.TimeoutExpired:
        print(f"Process {pid} did not terminate in time, forcing...")
        process.kill()
        return True
    except Exception as e:
        print(f"Error killing process {pid}: {str(e)}")
        return False

def kill_task_by_name(task_name):
    """通过任务名称终止进程"""
    process_info = load_process_info()
    killed = False
    
    for info in process_info:
        if info["task_name"] == task_name:
            print(f"Killing task '{task_name}' (PID: {info['pid']})...")
            if kill_process(info["pid"]):
                killed = True
                # 从进程信息中移除
                process_info.remove(info)
    
    if killed:
        # 更新进程信息文件
        with open(PROCESS_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(process_info, f, indent=4, ensure_ascii=False)
        print(f"Task '{task_name}' has been terminated.")
    else:
        print(f"No running task found with name '{task_name}'.")

def list_running_tasks():
    """列出所有正在运行的任务"""
    process_info = load_process_info()
    if not process_info:
        print("No running tasks found.")
        return
    
    print("\nCurrently running tasks:")
    print("-" * 80)
    print(f"{'Task Name':<30} {'PID':<10} {'Start Time':<20} {'Output File'}")
    print("-" * 80)
    
    for info in process_info:
        print(f"{info['task_name']:<30} {info['pid']:<10} {info['start_time']:<20} {info['output_file']}")

def main():
    parser = argparse.ArgumentParser(description='Manage running tasks')
    parser.add_argument('--list', action='store_true', help='List all running tasks')
    parser.add_argument('--kill', type=str, help='Kill task by name')
    parser.add_argument('--kill-all', action='store_true', help='Kill all running tasks')
    
    args = parser.parse_args()
    
    if args.list:
        list_running_tasks()
    elif args.kill:
        kill_task_by_name(args.kill)
    elif args.kill_all:
        process_info = load_process_info()
        for info in process_info:
            print(f"Killing task '{info['task_name']}' (PID: {info['pid']})...")
            kill_process(info['pid'])
        # 清空进程信息文件
        with open(PROCESS_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print("All tasks have been terminated.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 