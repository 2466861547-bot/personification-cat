"""
训练脚本: LLM LoRA 微调
支持自动检测硬件: Mac CPU / Mac MPS / GPU (V100 32GB / 4090 24GB)
支持断点续训: 自动检测或指定 checkpoint 恢复训练
支持优雅中断: Ctrl+C 自动保存当前状态
支持状态检查: --status 查看训练进度和 checkpoint

用法:
  # 自动检测硬件 (推荐)
  python scripts/train_llm.py --auto

  # 手动指定机器类型
  python scripts/train_llm.py --machine cpu          # Mac CPU
  python scripts/train_llm.py --machine mps          # Apple Silicon
  python scripts/train_llm.py --machine gpu_24g      # RTX 4090
  python scripts/train_llm.py --machine gpu_32g      # V100 32GB

  # 手动覆盖配置
  python scripts/train_llm.py --auto --model xxx --epochs 20

  # 断点续训: 自动检测最新 checkpoint
  python scripts/train_llm.py --auto --resume auto

  # 断点续训: 指定具体 checkpoint 路径
  python scripts/train_llm.py --auto --resume ./checkpoints/llm_mps/checkpoint-550

  # 检查训练状态
  python scripts/train_llm.py --auto --status
"""

import os

# 必须在任何项目模块导入之前设置，避免 whisper_finetune.py 的离线模式
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.llm_finetune import LLMFineTuner, LLMFineTuneConfig, get_machine_config, detect_device


def load_llm_data(data_file: str):
    """加载 LLM 训练数据"""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_idx = int(len(data) * 0.9)
    return data[:split_idx], data[split_idx:]


def main():
    parser = argparse.ArgumentParser(
        description="LLM LoRA 微调 - 宠物翻译 (支持自动检测硬件)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
机器类型:
  auto      自动检测硬件 (默认)
  cpu       Mac CPU / 无 GPU 环境
  mps       Apple Silicon M1/M2/M3
  gpu_8g    GPU 8GB (T4/P100)
  gpu_16g   GPU 16GB (V100 16GB/A10)
  gpu_24g   GPU 24GB (RTX 4090)
  gpu_32g   GPU 32GB (V100 32GB)
        """,
    )
    parser.add_argument("--data", default="./data/synthetic/llm_training_data.json",
                        help="训练数据路径")
    parser.add_argument("--machine", default="auto",
                        choices=["auto", "cpu", "mps", "gpu_8g", "gpu_16g", "gpu_24g", "gpu_32g"],
                        help="机器类型 (默认: auto)")
    parser.add_argument("--auto", action="store_true",
                        help="自动检测硬件 (等同于 --machine auto)")
    parser.add_argument("--model", default=None,
                        help="基础模型 (覆盖自动配置)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数 (覆盖自动配置)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="批大小 (覆盖自动配置)")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率 (覆盖自动配置)")
    parser.add_argument("--output", default=None,
                        help="输出目录 (覆盖自动配置)")
    parser.add_argument("--lora_r", type=int, default=None,
                        help="LoRA rank (覆盖自动配置)")
    parser.add_argument("--resume", default=None,
                        help="断点续训: auto (自动检测最近checkpoint) 或 指定checkpoint路径")
    parser.add_argument("--status", action="store_true",
                        help="检查训练状态和已有 checkpoint")
    parser.add_argument("--no-train", action="store_true",
                        help="跳过训练, 仅测试推理")
    parser.add_argument("--no-inference", action="store_true",
                        help="跳过推理测试")
    args = parser.parse_args()

    # ========== --auto 标志处理 ==========
    if args.auto:
        args.machine = "auto"

    # ========== 自动检测硬件 ==========
    if args.machine == "auto":
        device_info = detect_device()
        print(f"\n{'='*50}")
        print(f"🤖 硬件检测结果")
        print(f"{'='*50}")
        print(f"  设备类型: {device_info['device']}")
        print(f"  设备名称: {device_info['device_name']}")
        if device_info['gpu_available']:
            print(f"  GPU 数量: {device_info['gpu_count']}")
            print(f"  显存大小: {device_info['gpu_memory_gb']:.1f} GB")
        print(f"  操作系统: {device_info['platform']}")
        print(f"{'='*50}\n")

    # ========== 获取配置 ==========
    config = get_machine_config(args.machine)

    # ========== 手动覆盖配置 ==========
    if args.model:
        config.base_model = args.model
        print(f"🔄 覆盖模型: {args.model}")
    if args.epochs:
        config.num_train_epochs = args.epochs
        print(f"🔄 覆盖轮数: {args.epochs}")
    if args.batch_size:
        config.batch_size = args.batch_size
        print(f"🔄 覆盖批大小: {args.batch_size}")
    if args.lr:
        config.learning_rate = args.lr
        print(f"🔄 覆盖学习率: {args.lr}")
    if args.output:
        config.output_dir = args.output
        print(f"🔄 覆盖输出目录: {args.output}")
    if args.lora_r:
        config.lora_r = args.lora_r
        config.lora_alpha = args.lora_r * 2  # alpha = 2 * r
        print(f"🔄 覆盖 LoRA rank: {args.lora_r}")

    # ========== 检查训练状态 ==========
    if args.status:
        status = LLMFineTuner.check_training_status(config.output_dir)
        print(f"\n{'='*50}")
        print(f"📊 训练状态检查")
        print(f"{'='*50}")
        print(f"  输出目录: {status['output_dir']}")
        print(f"  目录存在: {'✅' if status['exists'] else '❌'}")

        if status["checkpoints"]:
            print(f"\n  📁 已有 Checkpoints ({len(status['checkpoints'])} 个):")
            for ckpt in status["checkpoints"]:
                step_info = f", step={ckpt.get('global_step', '?')}" if 'global_step' in ckpt else ""
                epoch_info = f", epoch={ckpt.get('epoch', '?')}" if 'epoch' in ckpt else ""
                latest = " ⭐ (最新)" if ckpt == status["latest_checkpoint"] else ""
                interrupted = " 🔴 (中断)" if "interrupted" in ckpt.get("name", "") else ""
                print(f"    • {ckpt['name']}{latest}{interrupted}")
                print(f"      路径: {ckpt['path']}")
                print(f"      时间: {ckpt['modified']}{step_info}{epoch_info}")
        else:
            print(f"\n  ⚠️  暂无 checkpoint")

        if status["final_exists"]:
            print(f"\n  ✅ 训练已完成 (存在 final/ 目录)")
        else:
            print(f"\n  🔄 训练未完成")

        if status["interrupted_info"]:
            info = status["interrupted_info"]
            print(f"\n  ⚠️  上次训练已中断:")
            print(f"    中断时间: {info.get('interrupted_at', 'unknown')}")
            print(f"    恢复命令: {info.get('resume_command', 'unknown')}")
            print(f"\n  💡 使用以下命令恢复训练:")
            print(f"    python scripts/train_llm.py --auto --resume auto")
        elif status["latest_checkpoint"]:
            print(f"\n  💡 使用以下命令恢复到最新 checkpoint:")
            print(f"    python scripts/train_llm.py --auto --resume auto")
            print(f"    或: python scripts/train_llm.py --auto --resume {status['latest_checkpoint']['path']}")

        print(f"{'='*50}\n")
        return

    # ========== 加载数据 ==========
    print(f"\n📂 加载训练数据: {args.data}")
    train_data, eval_data = load_llm_data(args.data)
    print(f"   训练集: {len(train_data)} 条")
    print(f"   验证集: {len(eval_data)} 条")

    # ========== 初始化微调器 ==========
    print(f"\n🔧 初始化微调器...")
    finetuner = LLMFineTuner(config)

    # ========== 训练 ==========
    if not args.no_train:
        print(f"\n{'='*50}")
        print(f"🚀 开始训练")
        print(f"{'='*50}")
        save_path = finetuner.train(train_data, eval_data, resume_from=args.resume)
        print(f"\n✅ 训练完成! LoRA adapter 保存到: {save_path}")

        # 推理测试
        if not args.no_inference:
            print(f"\n{'='*50}")
            print(f"🧪 测试推理")
            print(f"{'='*50}")
            test_input = (
                "宠物类型: cat\n"
                "品种: 橘猫\n"
                "声音特征: 短促的喵叫，重复2-3次，音调上扬\n"
                "情境: 早上刚起床\n"
                "请分析这只橘猫的情绪状态和需求。"
            )
            result = finetuner.inference(test_input)
            print(f"\n输入:\n{test_input}")
            print(f"\n输出:\n{result}")
    else:
        # 仅测试推理 (需要先加载已训练的模型)
        print("\n⚠️  --no-train 模式需要先训练模型")
        print("    请先运行训练, 或使用 load_finetuned() 加载已有模型")


if __name__ == "__main__":
    main()
