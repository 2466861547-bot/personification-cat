"""
训练脚本: Whisper 微调
用法: python scripts/train_whisper.py --data ./data/synthetic/whisper_labels.json
"""

import os
import sys
import json
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.whisper_finetune import WhisperFineTuner, WhisperFineTuneConfig
from src.audio.audio_processor import AudioProcessor


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
    parser.add_argument("--data", default="./data/synthetic/whisper_labels.json",
                        help="训练数据路径")
    parser.add_argument("--model", default="openai/whisper-large-v3",
                        help="基础模型")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="学习率")
    parser.add_argument("--output", default="./checkpoints/whisper", help="输出目录")
    args = parser.parse_args()

    # 加载数据
    print(f"加载训练数据: {args.data}")
    train_data, eval_data = load_whisper_data(args.data)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # 配置
    config = WhisperFineTuneConfig(
        model_name=args.model,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

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
