"""
训练脚本: Whisper 微调

模式:
  --quick    Mac 快速训练 (whisper-tiny, 5 epochs, 500 samples)
  --heavy    GPU 服务器重量级训练 (自动适配显存)
  默认       自定义训练

用法:
  # Mac 快速训练
  cd /Users/apple/Downloads/ai/personification-cat
  python scripts/train_whisper.py --quick

  # GPU 服务器重量级训练 (自动检测显存)
  python scripts/train_whisper.py --heavy

  # GPU 服务器指定显存
  python scripts/train_whisper.py --heavy --gpu-mem 24

  # 自定义训练
  python scripts/train_whisper.py --model openai/whisper-small --epochs 10 --batch_size 4
"""

import os
import sys
import json
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.whisper_finetune import (
    WhisperFineTuner, WhisperFineTuneConfig,
    get_mac_light_config, get_heavy_config,
    detect_device, get_gpu_memory_gb,
)


def load_whisper_data(label_file: str):
    """加载 Whisper 训练数据"""
    with open(label_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 转换为 Whisper 格式
    train_data = [{"audio_path": d["audio_path"], "text": d["text"]} for d in data]

    # 分割训练/验证
    split_idx = int(len(train_data) * 0.9)
    return train_data[:split_idx], train_data[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="Whisper 宠物声音识别微调")
    parser.add_argument("--quick", action="store_true",
                        help="Mac 快速训练模式 (whisper-tiny, 5 epochs, 500 samples)")
    parser.add_argument("--heavy", action="store_true",
                        help="GPU 服务器重量级训练模式")
    parser.add_argument("--gpu-mem", type=int, default=None,
                        help="GPU 显存大小 (GB), --heavy 模式下自动检测或手动指定")
    parser.add_argument("--data", default="./data/synthetic/whisper_labels.json",
                        help="训练数据路径")
    parser.add_argument("--model", default=None, help="基础模型")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="学习率")
    parser.add_argument("--max_samples", type=int, default=None, help="最大训练样本数")
    parser.add_argument("--output", default="./checkpoints/whisper", help="输出目录")
    args = parser.parse_args()

    # 检测设备
    device = detect_device()
    gpu_mem = get_gpu_memory_gb()
    print(f"\n🔍 设备检测:")
    print(f"   设备: {device}")
    if device == "cuda":
        print(f"   GPU 显存: {gpu_mem}GB")
    print()

    # 加载数据
    print(f"加载训练数据: {args.data}")
    train_data, eval_data = load_whisper_data(args.data)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # 选择配置
    if args.quick:
        print("\n🚀 Mac 快速训练模式 (--quick)")
        print("   模型: whisper-tiny | Epochs: 5 | Batch: 2 | 样本: 500")
        config = get_mac_light_config()

    elif args.heavy:
        print("\n🔥 GPU 服务器重量级训练模式 (--heavy)")
        # 确定显存
        mem = args.gpu_mem if args.gpu_mem else (gpu_mem if gpu_mem > 0 else 24)
        config = get_heavy_config(gpu_mem_gb=mem)
        print(f"   模型: {config.model_name} | Epochs: {config.num_train_epochs} | Batch: {config.batch_size}")
        print(f"   LoRA r: {config.lora_r} | 样本: 全量 | 显存: {mem}GB")

        # 如果没有 GPU, 警告
        if device == "mps":
            print("   ⚠ 未检测到 CUDA GPU, 将使用 MPS 训练 (速度较慢)")
            print("   建议: 在 Linux GPU 服务器上运行 --heavy 模式")
            # MPS 限制: 降级配置
            config.batch_size = 2
            config.num_train_epochs = 5
            config.lora_r = 8
            config.max_train_samples = 500
            config.model_name = "openai/whisper-tiny"
        elif device == "cpu":
            print("   ⚠ 未检测到 GPU, 将使用 CPU 训练 (非常慢)")
            print("   建议: 使用 Mac --quick 模式或 Linux GPU 服务器")
            config = get_mac_light_config()

    else:
        config = WhisperFineTuneConfig(
            model_name=args.model or "openai/whisper-tiny",
            num_train_epochs=args.epochs or 5,
            batch_size=args.batch_size or 2,
            learning_rate=args.lr or 1e-4,
            output_dir=args.output,
        )
        if args.max_samples:
            config.max_train_samples = args.max_samples

    print(f"\n📋 训练配置:")
    print(f"   模型: {config.model_name}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Batch: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   LoRA r: {config.lora_r}")
    print(f"   Max Samples: {config.max_train_samples or '全部'}")
    print(f"   Gradient Accumulation: {config.gradient_accumulation_steps}")
    print()

    # 初始化微调器
    finetuner = WhisperFineTuner(config)

    # 训练
    save_path = finetuner.train(train_data, eval_data)
    print(f"\n训练完成! 模型保存到: {save_path}")

    # 测试推理
    if eval_data:
        print("\n测试推理:")
        test_audio = eval_data[0]["audio_path"]
        expected = eval_data[0]["text"]
        result = finetuner.inference(test_audio)
        print(f"  期望: {expected}")
        print(f"  实际: {result}")


if __name__ == "__main__":
    main()
