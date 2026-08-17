"""
推理脚本: 端到端宠物翻译
用法:
  python scripts/inference.py --mode pet_to_text --audio ./data/raw/cat_sounds/test.wav
  python scripts/inference.py --mode text_to_pet --text "过来吃饭" --pet cat
  python scripts/inference.py --mode chat
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import TranslationPipeline


def main():
    parser = argparse.ArgumentParser(description="宠物翻译推理")
    parser.add_argument("--mode", required=True,
                        choices=["pet_to_text", "text_to_pet", "chat"],
                        help="运行模式")
    parser.add_argument("--audio", default="", help="宠物声音文件路径")
    parser.add_argument("--text", default="", help="人类语言文本")
    parser.add_argument("--pet", default="cat", choices=["cat", "dog"], help="宠物类型")
    parser.add_argument("--breed", default="通用", help="品种")
    parser.add_argument("--output", default="", help="输出音频路径")
    parser.add_argument("--whisper_model", default="", help="Whisper 微调模型路径")
    parser.add_argument("--llm_adapter", default="", help="LLM LoRA adapter 路径")
    args = parser.parse_args()

    # 初始化 Pipeline
    pipeline = TranslationPipeline(
        whisper_model_path=args.whisper_model if args.whisper_model else None,
        llm_adapter_path=args.llm_adapter if args.llm_adapter else None,
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
        print(f"\n分析:\n{result.get('analysis')}")

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
