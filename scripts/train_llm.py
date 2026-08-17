"""
训练脚本: LLM LoRA 微调
用法: python scripts/train_llm.py --data ./data/synthetic/llm_training_data.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.llm_finetune import LLMFineTuner, LLMFineTuneConfig


def load_llm_data(data_file: str):
    """加载 LLM 训练数据"""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_idx = int(len(data) * 0.9)
    return data[:split_idx], data[split_idx:]


def main():
    parser = argparse.ArgumentParser(description="LLM LoRA 微调 - 宠物翻译")
    parser.add_argument("--data", default="./data/synthetic/llm_training_data.json",
                        help="训练数据路径")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="基础模型")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--output", default="./checkpoints/llm", help="输出目录")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    args = parser.parse_args()

    # 加载数据
    print(f"加载训练数据: {args.data}")
    train_data, eval_data = load_llm_data(args.data)
    print(f"训练集: {len(train_data)} 条, 验证集: {len(eval_data)} 条")

    # 配置
    config = LLMFineTuneConfig(
        base_model=args.model,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
        lora_r=args.lora_r,
    )

    # 初始化微调器
    finetuner = LLMFineTuner(config)

    # 训练
    save_path = finetuner.train(train_data, eval_data)
    print(f"\n训练完成! LoRA adapter 保存到: {save_path}")

    # 测试推理
    print("\n测试推理:")
    test_input = (
        "宠物类型: cat\n"
        "品种: 橘猫\n"
        "声音特征: 短促的喵叫，重复2-3次，音调上扬\n"
        "情境: 早上刚起床\n"
        "请分析这只橘猫的情绪状态和需求。"
    )
    result = finetuner.inference(test_input)
    print(f"输入:\n{test_input}")
    print(f"\n输出:\n{result}")


if __name__ == "__main__":
    main()
