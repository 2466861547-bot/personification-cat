"""
推理脚本: 端到端宠物翻译
支持自动检测 LLM / Whisper 最新 checkpoint / final
用法:
  python scripts/inference.py --mode pet_to_text --audio ./data/raw/cat_sounds/test.wav
  python scripts/inference.py --mode text_to_pet --text "过来吃饭" --pet cat
  python scripts/inference.py --mode chat

自动检测:
  不传 --llm_adapter / --whisper_model 时自动查找 ./checkpoints/ 下最新的模型
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import TranslationPipeline


def find_latest_adapter(checkpoints_dir: str, model_prefix: str) -> str:
    """查找 checkpoints 目录下最新的 adapter (优先 final, 其次最新 checkpoint)

    Args:
        checkpoints_dir: checkpoints 根目录 (如 ./checkpoints)
        model_prefix: 模型前缀 (如 'llm' 或 'whisper')

    Returns:
        最新 adapter 路径，未找到返回空字符串
    """
    candidates = []

    # 遍历所有匹配的模型目录
    for subdir in sorted(glob.glob(os.path.join(checkpoints_dir, f"{model_prefix}_*"))):
        if not os.path.isdir(subdir):
            continue

        # 1. 优先查找 final 目录
        final_dir = os.path.join(subdir, "final")
        if os.path.isdir(final_dir):
            candidates.append((final_dir, os.path.getmtime(final_dir)))

        # 2. 查找中断保存的 checkpoint
        interrupted_dir = os.path.join(subdir, "checkpoint-interrupted")
        if os.path.isdir(interrupted_dir):
            candidates.append((interrupted_dir, os.path.getmtime(interrupted_dir)))

        # 3. 查找最新的 checkpoint-*
        ckpt_dirs = glob.glob(os.path.join(subdir, "checkpoint-*"))
        for ckpt_dir in ckpt_dirs:
            if os.path.isdir(ckpt_dir) and not ckpt_dir.endswith("checkpoint-interrupted"):
                candidates.append((ckpt_dir, os.path.getmtime(ckpt_dir)))

    if not candidates:
        return ""

    # 按修改时间排序，最新的在前
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def auto_detect_models():
    """自动检测可用的 LLM 和 Whisper 模型路径"""
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    models = {}

    # 检测 LLM adapter
    llm_path = find_latest_adapter(base_dir, "llm")
    if llm_path:
        models["llm_adapter"] = llm_path

    # 检测 Whisper model
    whisper_path = find_latest_adapter(base_dir, "whisper")
    if whisper_path:
        models["whisper_model"] = whisper_path

    return models


def main():
    parser = argparse.ArgumentParser(description="宠物翻译推理")
    parser.add_argument("--mode", required=False,
                        choices=["pet_to_text", "text_to_pet", "chat"],
                        help="运行模式")
    parser.add_argument("--audio", default="", help="宠物声音文件路径")
    parser.add_argument("--text", default="", help="人类语言文本")
    parser.add_argument("--pet", default="cat", choices=["cat", "dog"], help="宠物类型")
    parser.add_argument("--breed", default="通用", help="品种")
    parser.add_argument("--output", default="", help="输出音频路径")
    parser.add_argument("--whisper_model", default="", help="Whisper 微调模型路径 (不指定则自动检测)")
    parser.add_argument("--llm_adapter", default="", help="LLM LoRA adapter 路径 (不指定则自动检测)")
    parser.add_argument("--embedding-model", default="",
                        help="Embedding 模型名 (覆盖默认，MPS 默认 BAAI/bge-small-zh-v1.5, GPU 默认 BAAI/bge-large-zh-v1.5)")
    parser.add_argument("--no-rag", action="store_true",
                        help="禁用 RAG 检索 (无需 Embedding 模型)")
    parser.add_argument("--list-models", action="store_true",
                        help="列出可用的模型 (自动检测结果)")
    args = parser.parse_args()

    # --list-models: 列出可用模型并退出
    if args.list_models:
        models = auto_detect_models()
        print("\n" + "=" * 50)
        print("📦 自动检测到的可用模型")
        print("=" * 50)
        if "llm_adapter" in models:
            print(f"  ✅ LLM adapter: {models['llm_adapter']}")
        else:
            print(f"  ⚠️  未找到 LLM adapter (运行训练: python scripts/train_llm.py --auto)")
        if "whisper_model" in models:
            print(f"  ✅ Whisper model: {models['whisper_model']}")
        else:
            print(f"  ⚠️  未找到 Whisper model (运行训练: python scripts/train_whisper.py)")
        print("=" * 50 + "\n")
        return

    if not args.mode:
        parser.error("--mode is required (choices: pet_to_text, text_to_pet, chat) or use --list-models")

    # ============ 用户交互层开始 ============
    mode_names = {
        "pet_to_text": "宠物声音 → 人类语言 (声音输入)",
        "text_to_pet": "人类语言 → 宠物声音 (文字输入)",
        "chat": "宠物翻译对话 (对话模式)",
    }
    print("\n" + "=" * 60)
    print(f"🏠 【用户交互层】{mode_names.get(args.mode, args.mode)}")
    print("=" * 60)

    if args.mode == "pet_to_text":
        print(f"  📥 声音输入模式")
        print(f"     音频文件: {args.audio}")
        print(f"     宠物类型: {args.pet}")
        print(f"     品种: {args.breed}")
    elif args.mode == "text_to_pet":
        print(f"  📥 文字输入模式")
        print(f"     输入文本: '{args.text}'")
        print(f"     目标宠物: {args.pet}")
        print(f"     输出路径: {args.output or './output/'}")
    elif args.mode == "chat":
        print(f"  💬 对话模式")
        print(f"     退出命令: quit / exit / q")
    print(f"  {'─'*60}")
    # ============ 用户交互层结束 ============

    # 自动检测模型 (未手动指定时)
    auto_models = auto_detect_models()
    whisper_path = args.whisper_model or auto_models.get("whisper_model", "")
    llm_path = args.llm_adapter or auto_models.get("llm_adapter", "")

    if not args.whisper_model and whisper_path:
        print(f"💡 自动检测到 Whisper 模型: {whisper_path}")
    if not args.llm_adapter and llm_path:
        print(f"💡 自动检测到 LLM adapter: {llm_path}")

    if not whisper_path and not llm_path:
        print("⚠️  未检测到任何已训练模型，将使用基础模型 (性能有限)")
        print("   训练命令:")
        print("     python scripts/train_whisper.py")
        print("     python scripts/train_llm.py --auto")
        print("   查看可用模型: python scripts/inference.py --list-models")

    # 初始化 Pipeline
    pipeline = TranslationPipeline(
        whisper_model_path=whisper_path if whisper_path else None,
        llm_adapter_path=llm_path if llm_path else None,
        embedding_model=args.embedding_model if args.embedding_model else None,
        use_rag=not args.no_rag,
    )

    if args.mode == "pet_to_text":
        # 宠物声音 → 人类语言
        result = pipeline.pet_sound_to_text(
            audio_path=args.audio,
            pet_type=args.pet,
            breed=args.breed,
        )
        print("\n" + "=" * 50)
        print("宠物声音分析结果")
        print("=" * 50)
        print(f"输入音频: {result.get('input_audio')}")
        print(f"Whisper 识别: {result.get('whisper_transcription')}")

        # 打印 RAG 检索详情
        rag_details = result.get("rag_details", [])
        if rag_details:
            print(f"\n📚 知识库检索结果 ({len(rag_details)} 条):")
            print(f"{'-'*40}")
            for i, r in enumerate(rag_details, 1):
                meta = r.get("metadata", {})
                score = r.get("score", 0)
                text = r.get("text", "")
                print(f"  [{i}] 相关度: {score:.3f}")
                print(f"      类型: {meta.get('pet_type','')} | 品种: {meta.get('breed','')} | 情绪: {meta.get('emotion','')}")
                print(f"      内容: {text[:100]}{'...' if len(text) > 100 else ''}")
                if meta.get("meaning"):
                    print(f"      含义: {meta['meaning']}")
                if meta.get("advice"):
                    print(f"      建议: {meta['advice']}")
                print()

        print(f"\n🤖 LLM 分析结论:")
        print(f"{'='*50}")
        print(f"{result.get('analysis')}")

    elif args.mode == "text_to_pet":
        # 人类语言 → 宠物声音
        result = pipeline.human_text_to_pet_sound(
            text=args.text,
            target_pet=args.pet,
            output_path=args.output,
        )
        print("\n" + "=" * 50)
        print("宠物声音生成结果")
        print("=" * 50)
        print(f"输入文本: {result.get('input_text')}")
        print(f"目标宠物: {result.get('target_pet')}")
        print(f"状态: {result.get('status')}")
        if result.get("output_path"):
            print(f"输出音频: {result.get('output_path')}")

    elif args.mode == "chat":
        # 对话模式
        print("=" * 50)
        print("喵汪翻译官 - 宠物翻译对话")
        print("输入 'quit' 退出")
        print("=" * 50)

        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            if not user_input:
                continue

            response = pipeline.chat(user_input)
            print(f"\n喵汪翻译官: {response}")


if __name__ == "__main__":
    main()
