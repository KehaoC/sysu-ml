import torch
from datetime import datetime
from experiment import run_experiments

if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建带时间戳的输出目录
    output_dir = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 运行所有实验并生成可视化
    results = run_experiments(output_dir) 