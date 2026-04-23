"""
模型对比脚本 - 依次运行FNN、RNN、CNN、Transformer四个模型并生成对比报告
"""

import subprocess
import sys
import time
from datetime import datetime


def run_demo(demo_name, demo_file):
    """运行单个demo并返回执行结果"""
    print("\n" + "=" * 80)
    print(f"🚀 开始运行 {demo_name}")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, demo_file],
            capture_output=False,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ {demo_name} 运行成功! 耗时: {elapsed_time:.2f}秒")
            return True, elapsed_time
        else:
            print(f"\n❌ {demo_name} 运行失败!")
            return False, elapsed_time
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {demo_name} 运行超时!")
        return False, time.time() - start_time
    except Exception as e:
        print(f"\n❌ {demo_name} 出现异常: {str(e)}")
        return False, time.time() - start_time


def main():
    print("=" * 80)
    print("深度学习模型对比实验")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n本脚本将依次运行以下四个模型:")
    print("  1. FNN (前馈神经网络)")
    print("  2. RNN (循环神经网络/LSTM)")
    print("  3. CNN (卷积神经网络)")
    print("  4. Transformer (Vision Transformer) ⭐ 新增")
    print("\n每个模型将训练多个epoch，请等待完成...")
    print("=" * 80)
    
    demos = [
        ("FNN - 前馈神经网络", "fnn_demo.py"),
        ("RNN - 循环神经网络", "rnn_demo.py"),
        ("CNN - 卷积神经网络", "cnn_demo.py"),
        ("Transformer - Vision Transformer", "transformer_demo.py"),
    ]
    
    results = []
    total_start = time.time()
    
    for demo_name, demo_file in demos:
        success, elapsed_time = run_demo(demo_name, demo_file)
        results.append({
            'name': demo_name,
            'file': demo_file,
            'success': success,
            'time': elapsed_time
        })
        
        if not success:
            print(f"\n⚠️  {demo_name} 运行失败，是否继续? (y/n): ", end='')
            choice = input().strip().lower()
            if choice != 'y':
                break
    
    total_elapsed = time.time() - total_start
    
    # 打印总结报告
    print("\n" + "=" * 80)
    print("📊 实验总结报告")
    print("=" * 80)
    print(f"总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n各模型运行情况:")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"{i}. {result['name']}")
        print(f"   状态: {status}")
        print(f"   耗时: {result['time']:.2f}秒 ({result['time']/60:.2f}分钟)")
        print()
    
    print("=" * 80)
    print("📁 生成的文件:")
    print("=" * 80)
    
    generated_files = [
        "FNN模型:",
        "  - fnn_mnist.pth",
        "  - fnn_training_curve.png",
        "  - fnn_predictions.png",
        "",
        "RNN模型:",
        "  - rnn_mnist.pth",
        "  - rnn_training_curve.png",
        "  - rnn_predictions.png",
        "",
        "CNN模型:",
        "  - cnn_mnist.pth",
        "  - cnn_training_curve.png",
        "  - cnn_predictions.png",
        "  - cnn_filters.png",
        "",
        "Transformer模型: ⭐",
        "  - transformer_mnist.pth",
        "  - transformer_training_curve.png",
        "  - transformer_predictions.png",
        "  - transformer_attention.png (注意力可视化)",
    ]
    
    for file_info in generated_files:
        print(file_info)
    
    print("\n" + "=" * 80)
    print("💡 提示:")
    print("=" * 80)
    print("• 查看 MODELS_README.md 了解各模型的详细说明")
    print("• 对比 training_curve.png 观察不同模型的训练效果")
    print("• 查看 predictions.png 了解模型的预测能力")
    print("• ⭐ 特别关注 transformer_attention.png 理解Self-Attention机制")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断实验")
        sys.exit(0)
