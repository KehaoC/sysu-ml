import time
import random
import sys

def print_progress_bar(iteration, total, length=40):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '▓' * filled_length + '░' * (length - filled_length)
    sys.stdout.write(f'\r\033[32m|{bar}| {percent:.2f}% Complete\033[0m')  # 使用绿色文本
    sys.stdout.flush()

def random_print():
    messages = [
        "正在准备数据集...",
        "加载训练模型...",
        "连接到训练服务器...",
        "处理训练请求...",
        "检索训练参数...",
        "执行训练步骤...",
        "更新训练状态...",
        "验证模型准确性...",
        "清理训练缓存...",
        "同步训练数据...",
        "模型训练完成！",
        "模型加载成功！",
        "服务器连接成功！",
        "请求处理完毕！",
        "参数检索成功！",
        "训练步骤执行成功！",
        "状态更新成功！",
        "模型验证成功！",
        "缓存清理完成！",
        "数据同步完成！"
    ]
    print(f"\033[34m{random.choice(messages)}\033[0m")  # 使用蓝色文本

def simulate_large_program():
    total_steps = 40# 设置为 120 步，以便在 1 分钟内完成
    for i in range(total_steps + 1):
        if random.choice([True, False]):  # 随机决定是否打印进度条
            print_progress_bar(i, total_steps)
        else:
            random_print()  # 随机打印信息

        time.sleep(random.uniform(0.05, 0.15))  # 每步处理时间为 0.4 到 0.6 秒

        if i % 10 == 0 and i != 0:
            print(f"\033[35mProcessing step {i}...\033[0m")  # 使用紫色文本
            time.sleep(random.uniform(0.05, 0.15))  # 随机额外处理时间

        if i == total_steps:
            print("\n\033[32m任务完成！\033[0m")  # 使用绿色文本

if __name__ == "__main__":
    simulate_large_program()
