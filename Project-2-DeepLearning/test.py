import torch
import platform
import sys

def test_pytorch_environment():
    print("\n=== PyTorch Environment Test ===\n")
    
    # 基本系统信息
    print("System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA 可用性
    print("\nCUDA Support:")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    
    # MPS (Metal Performance Shaders) 可用性
    print("\nMPS Support:")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")
    
    # 确定最佳可用设备
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"\nBest Available Device: {device}")
    
    # 简单的张量运算测试
    print("\nTensor Operations Test:")
    try:
        # 创建测试张量
        x = torch.randn(1000, 1000)
        if device != "cpu":
            x = x.to(device)
        
        # 进行一些基本运算
        start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        
        if device == "cuda":
            start_time.record()
        
        # 执行一些计算密集型操作
        for _ in range(100):
            y = torch.matmul(x, x)
        
        if device == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            print(f"CUDA Computation Time: {start_time.elapsed_time(end_time):.2f} ms")
        
        print("Tensor operations completed successfully!")
        
    except Exception as e:
        print(f"Error during tensor operations: {str(e)}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_pytorch_environment()