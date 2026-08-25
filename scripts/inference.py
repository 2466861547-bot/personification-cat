"""
推理脚本: 端到端宠物翻译 + 声纹克隆
支持自动检测 LLM / Whisper 最新 checkpoint / final
用法:
  python scripts/inference.py --mode pet_to_text --audio ./data/raw/cat_sounds/test.wav
  python scripts/inference.py --mode text_to_pet --text "过来吃饭" --pet cat
  python scripts/inference.py --mode human_to_human --audio ./input/voice.wav --target-voice "林志玲"
  python scripts/inference.py --mode human_to_pet --audio ./input/voice.wav --pet cat
  python scripts/inference.py --mode chat

自动检测:
  不传 --llm_adapter / --whisper_model 时自动查找 ./checkpoints/ 下最新的模型
"""

import os
import sys
import glob
import argparse
import subprocess


COSYVOICE_DEPS = [
    ("modelscope", "modelscope>=1.18.0"),
    ("safetensors", "safetensors>=0.4.0"),
    ("librosa", "librosa>=0.10.2"),
    ("yaml", "pyyaml>=6.0.2"),
    ("sentencepiece", "sentencepiece>=0.1.99"),
]


def _ensure_cosyvoice_deps():
    """自动安装 CosyVoice 所需依赖"""
    missing = []
    for module_name, pip_spec in COSYVOICE_DEPS:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_spec)

    if missing:
        print(f"  📦 CosyVoice 缺少依赖，正在自动安装: {', '.join(missing)}")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  ✅ 依赖安装完成")
        except subprocess.CalledProcessError:
            print(f"  ⚠️ 自动安装失败，请手动运行: pip install {' '.join(missing)}")
            return False
    return True


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import TranslationPipeline
from src.models.audio_generation import AudioGenerator, HUMAN_VOICE_PRESETS


def list_available_voices():
    """列出所有可用的人类声线预设 (Bark 内置 + CosyVoice 预置声线)"""
    print("\n" + "=" * 50)
    print("🎤 可用的人类声线")
    print("=" * 50)

    # Bark 内置声线
    print("\n📦 Bark 内置声线预设:")
    for name, preset in HUMAN_VOICE_PRESETS.items():
        if name == "default":
            print(f"  (默认) {name}: {preset}")
        else:
            print(f"  ✅ {name}: {preset}")

    # CosyVoice 预置声线 - 通过类方法自动注册 (不加载模型)
    from src.models.audio_generation import AudioGenerator
    AudioGenerator._register_builtin_voices()
    cosyvoice_voices = AudioGenerator.PRESET_VOICE_REFS
    if cosyvoice_voices:
        print("\n🎯 CosyVoice 真实声纹克隆预置 (克隆目标人物声音):")
        for name, path in cosyvoice_voices.items():
            exists = "✅" if os.path.isfile(path) else "⚠️"
            print(f"  {exists} {name}: {path}")
    else:
        print("\n🎯 CosyVoice 预置声线: (暂无)")

    print("\n" + "=" * 50)
    print("提示:")
    print("  1. 使用 --reference-audio 指定自定义参考音频进行克隆")
    print("  2. 使用 --reference-text 指定参考音频文字 (更精准)")
    print("  3. 直接用 --target-voice '林志玲' 即可触发 CosyVoice 克隆")
    print("=" * 50 + "\n")


def find_latest_adapter(checkpoints_dir: str, model_prefix: str) -> str:
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
    parser = argparse.ArgumentParser(description="宠物翻译推理 + 声纹克隆")
    parser.add_argument("--mode", required=False,
                        choices=["pet_to_text", "text_to_pet", "human_to_human", "human_to_pet", "pet_to_human_voice", "chat"],
                        help="运行模式")
    parser.add_argument("--audio", default="", help="音频文件路径 (宠物或人类声音)")
    parser.add_argument("--text", default="", help="人类语言文本")
    parser.add_argument("--pet", default="cat", choices=["cat", "dog"], help="宠物类型")
    parser.add_argument("--breed", default="通用", help="品种")
    parser.add_argument("--target-voice", default="default",
                        help="目标人类声线 (如 '林志玲', '檀健次', '温柔女声', '磁性男声') 或自定义声纹文件 (.json/.npz)")
    parser.add_argument("--reference-audio", default="",
                        help="参考音频路径 (3-15秒 .wav)，用于 CosyVoice 真实声纹克隆 (如林志玲的声音样本)")
    parser.add_argument("--reference-text", default="",
                        help="参考音频的文字内容，用于 CosyVoice zero-shot 克隆 (留空则使用 cross-lingual 模式)")
    parser.add_argument("--list-voices", action="store_true",
                        help="列出所有可用的人类声线预设")
    parser.add_argument("--output", default="", help="输出音频路径")
    parser.add_argument("--whisper_model", default="", help="Whisper 微调模型路径 (不指定则自动检测)")
    parser.add_argument("--llm_adapter", default="", help="LLM LoRA adapter 路径 (不指定则自动检测)")
    parser.add_argument("--embedding-model", default="",
                        help="Embedding 模型名 (覆盖默认，MPS 默认 BAAI/bge-small-zh-v1.5, GPU 默认 BAAI/bge-large-zh-v1.5)")
    parser.add_argument("--no-rag", action="store_true",
                        help="禁用 RAG 检索 (无需 Embedding 模型)")
    parser.add_argument("--no-llm-polish", action="store_true",
                        help="禁用 LLM 润色 (仅使用模板生成拟人化文本)")
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

    if args.list_voices:
        list_available_voices()
        return

    if not args.mode:
        parser.error("--mode is required (choices: pet_to_text, text_to_pet, human_to_human, human_to_pet, pet_to_human_voice, chat) or use --list-models / --list-voices")

    # ============ 用户交互层开始 ============
    mode_names = {
        "pet_to_text": "宠物声音 → 人类语言 (声音输入)",
        "text_to_pet": "人类语言 → 宠物声音 (文字输入)",
        "human_to_human": "人类声音 → 定制人类声音 (声纹克隆)",
        "human_to_pet": "人类声音 → 宠物声音 (声音输入)",
        "pet_to_human_voice": "宠物声音 → 人类声音 (声纹克隆)",
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
    elif args.mode == "human_to_human":
        print(f"  📥 人类声音输入 → 声纹克隆输出")
        print(f"     输入音频: {args.audio}")
        print(f"     目标声线: {args.target_voice}")
        if args.reference_audio:
            print(f"     参考音频: {args.reference_audio}")
            if args.reference_text:
                print(f"     参考文本: {args.reference_text}")
        print(f"     输出路径: {args.output or './output/'}")
    elif args.mode == "human_to_pet":
        print(f"  📥 人类声音输入 → 宠物声音输出")
        print(f"     输入音频: {args.audio}")
        print(f"     目标宠物: {args.pet}")
        print(f"     输出路径: {args.output or './output/'}")
    elif args.mode == "pet_to_human_voice":
        print(f"  📥 宠物声音输入 → 人类声音 (声纹克隆) 输出")
        print(f"     输入音频: {args.audio}")
        print(f"     宠物类型: {args.pet}")
        print(f"     目标声线: {args.target_voice}")
        if args.reference_audio:
            print(f"     参考音频: {args.reference_audio}")
            if args.reference_text:
                print(f"     参考文本: {args.reference_text}")
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

    elif args.mode == "human_to_human":
        # 人类声音 → 人类声音 (声纹克隆)
        _ensure_cosyvoice_deps()
        result = pipeline.human_sound_to_human_sound(
            audio_path=args.audio,
            target_voice=args.target_voice,
            output_path=args.output,
            reference_audio=args.reference_audio or None,
            reference_text=args.reference_text or None,
        )
        print("\n" + "=" * 50)
        print("声纹克隆结果")
        print("=" * 50)
        print(f"输入音频: {result.get('input_audio')}")
        print(f"识别文本: {result.get('recognized_text')}")
        if result.get("llm_analysis"):
            print(f"LLM 分析: {result.get('llm_analysis')[:150]}...")
        print(f"目标声线: {result.get('target_voice')}")
        print(f"状态: {result.get('status')}")
        if result.get("output_path"):
            print(f"输出音频: {result.get('output_path')}")
        if result.get("note"):
            print(f"说明: {result.get('note')}")

    elif args.mode == "human_to_pet":
        # 人类声音 → 宠物声音
        result = pipeline.human_sound_to_pet_sound(
            audio_path=args.audio,
            target_pet=args.pet,
            output_path=args.output,
        )
        print("\n" + "=" * 50)
        print("人类声音 → 宠物声音转换结果")
        print("=" * 50)
        print(f"输入音频: {result.get('input_audio')}")
        print(f"识别文本: {result.get('recognized_text')}")
        print(f"目标宠物: {result.get('target_pet')}")
        print(f"状态: {result.get('status')}")
        if result.get("output_path"):
            print(f"输出音频: {result.get('output_path')}")
        if result.get("note"):
            print(f"说明: {result.get('note')}")

    elif args.mode == "pet_to_human_voice":
        # 宠物声音 → 人类声音 (声纹克隆)
        _ensure_cosyvoice_deps()
        result = pipeline.pet_sound_to_human_sound(
            audio_path=args.audio,
            target_voice=args.target_voice,
            pet_type=args.pet,
            breed=args.breed,
            output_path=args.output,
            use_llm_polish=not args.no_llm_polish,
            reference_audio=args.reference_audio or None,
            reference_text=args.reference_text or None,
        )
        print("\n" + "=" * 50)
        print("宠物声音 → 人类声音 (声纹克隆) 结果")
        print("=" * 50)
        print(f"输入音频: {result.get('input_audio')}")
        if result.get("emotion"):
            print(f"情绪识别: {result.get('emotion')} (置信度: {result.get('emotion_confidence', 0):.2f})")
        if result.get("description"):
            print(f"情绪描述: {result.get('description')}")
        if result.get("human_text"):
            source = result.get("human_text_source", "unknown")
            source_label = "LLM润色" if source == "llm_polished" else "模板生成"
            print(f"拟人化文本 [{source_label}]: {result.get('human_text')[:150]}...")
        print(f"目标声线: {result.get('target_voice')}")
        print(f"状态: {result.get('status')}")
        if result.get("output_path"):
            print(f"输出音频: {result.get('output_path')}")
        if result.get("note"):
            print(f"说明: {result.get('note')}")

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
