"""
GPU性能基准测试 - 对比CPU和GPU的性能差异
"""

import torch
import time
import matplotlib.pyplot as plt

def benchmark_device(device_name, device):
    """测试指定设备的性能"""
    print(f"\n{'='*60}")
    print(f"测试设备: {device_name}")
    print(f"{'='*60}")
    
    sizes = [1000, 2000, 5000, 10000]
    times = []
    
    for size in sizes:
        # 创建随机矩阵
        if 'cuda' in str(device):
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)
        else:
            A = torch.randn(size, size)
            B = torch.randn(size, size)
        
        # 预热
        if size == sizes[0]:
            for _ in range(5):
                C = torch.mm(A, B)
        
        # 正式测试
        start_time = time.time()
        iterations = 100 if size < 5000 else 10
        
        for _ in range(iterations):
            C = torch.mm(A, B)
        
        # 同步CUDA操作
        if 'cuda' in str(device):
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start_time) / iterations * 1000  # 毫秒
        times.append(elapsed)
        
        print(f"  矩阵大小: {size}x{size:5d} | "
              f"平均时间: {elapsed:7.2f}ms | "
              f"GFLOPS: {2 * size**3 / iterations / (elapsed/1000) / 1e9:.2f}")
    
    return sizes, times

def main():
    print("\n" + "="*60)
    print("🚀 PyTorch CPU vs GPU 性能对比测试")
    print("="*60)
    
    # 检查可用设备
    devices = []
    
    # 测试CPU
    devices.append(("CPU", torch.device('cpu')))
    
    # 测试GPU
    if torch.cuda.is_available():
        devices.append(("GPU (RTX 4070)", torch.device('cuda:0')))
    elif torch.backends.mps.is_available():
        devices.append(("MPS (Apple Silicon)", torch.device('mps')))
    else:
        print("\n⚠️  未检测到GPU,仅测试CPU")
    
    results = {}
    for name, device in devices:
        try:
            sizes, times = benchmark_device(name, device)
            results[name] = {'sizes': sizes, 'times': times}
        except Exception as e:
            print(f"\n❌ {name} 测试失败: {e}")
    
    # 可视化结果
    if len(results) > 0:
        plot_results(results)
    
    # 总结
    print("\n" + "="*60)
    print("📊 性能总结")
    print("="*60)
    
    if len(results) > 1:
        names = list(results.keys())
        cpu_times = results[names[0]]['times']
        
        for name in names[1:]:
            gpu_times = results[name]['times']
            speedup = [c/g for c, g in zip(cpu_times, gpu_times)]
            avg_speedup = sum(speedup) / len(speedup)
            
            print(f"\n{name} 相比 CPU:")
            print(f"  平均加速比: {avg_speedup:.1f}x")
            print(f"  最大加速比: {max(speedup):.1f}x")
            print(f"  最小加速比: {min(speedup):.1f}x")
    
    print("\n✅ 基准测试完成!")

def plot_results(results):
    """绘制性能对比图"""
    plt.figure(figsize=(12, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        plt.plot(data['sizes'], data['times'], 
                marker='o', linewidth=2, markersize=8,
                label=name, color=color)
    
    plt.xlabel('矩阵大小', fontsize=12, fontweight='bold')
    plt.ylabel('平均时间 (ms)', fontsize=12, fontweight='bold')
    plt.title('CPU vs GPU 矩阵乘法性能对比', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    
    # 添加网格标签
    plt.xticks(results[list(results.keys())[0]]['sizes'], 
               [str(s) for s in results[list(results.keys())[0]]['sizes']])
    
    plt.tight_layout()
    plt.savefig('benchmark_result.png', dpi=150, bbox_inches='tight')
    print("\n📊 性能对比图已保存: benchmark_result.png")

if __name__ == "__main__":
    main()
