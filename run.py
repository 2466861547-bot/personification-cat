#!/usr/bin/env python3
"""
personification-cat 全功能本地运行脚本
菜单驱动,一键体验所有功能
"""

import os
import sys
import time
import json
import shutil

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
ENTERTAINMENT_DIR = os.path.join(OUTPUT_DIR, "entertainment")
TRANSLATION_DIR = os.path.join(OUTPUT_DIR, "translation")

os.makedirs(ENTERTAINMENT_DIR, exist_ok=True)
os.makedirs(TRANSLATION_DIR, exist_ok=True)


def print_banner():
    """打印横幅"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   personification-cat  宠物声音与人声双向翻译 + 逗宠娱乐    ║
║                                                              ║
║   Whisper 微调 | LLM LoRA | RAG 检索 | LangChain Agent      ║
║   多智能体协作 | 定时调度 | 相声/唱歌/安抚/逗乐/讲故事      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """打印主菜单"""
    print("""
┌─────────────────────────────────────────────────────────────┐
│                      主功能菜单                              │
├─────────────────────────────────────────────────────────────┤
│  【数据与训练】                                              │
│   1. 生成训练数据 (合成音频 + LLM指令数据)                  │
│   2. 训练 Whisper 模型 (宠物声音 ASR)                       │
│   3. 训练 LLM 模型 (情绪理解 + 翻译)                       │
│                                                              │
│  【声音翻译】                                                │
│   4. 宠物声音 → 人类语言                                    │
│   5. 人类语言 → 宠物声音                                    │
│   6. 对话模式                                                │
│                                                              │
│  【逗宠娱乐】                                                │
│   7. 逗宠相声                                                │
│   8. 唱歌逗宠                                                │
│   9. 安抚音                                                  │
│  10. 逗乐音                                                  │
│  11. 讲故事                                                  │
│  12. 查看所有娱乐模式                                       │
│                                                              │
│  【定时与自动化】                                            │
│  13. 启动定时逗宠 (3-5次/天)                                │
│  14. 多智能体模式 (自动编排)                                │
│  15. 查看系统状态                                            │
│                                                              │
│  【模型优化】                                                │
│  16. Whisper 优化推理 (CTranslate2 加速)                    │
│  17. LLM vLLM 推理 (AWQ 量化)                              │
│  18. RAG 混合检索测试                                       │
│                                                              │
│  0. 退出                                                     │
└─────────────────────────────────────────────────────────────┘
    """)


def get_input(prompt, default=""):
    """获取用户输入"""
    val = input(prompt).strip()
    return val if val else default


def get_pet_type():
    """选择宠物类型"""
    while True:
        choice = input("选择宠物类型 [1]猫  [2]狗 (默认: 猫): ").strip()
        if choice == "2":
            return "dog"
        return "cat"


def check_file(path):
    """检查文件是否存在"""
    if not os.path.exists(path):
        print(f"  [错误] 文件不存在: {path}")
        return False
    return True


# =============================================
# 1. 生成训练数据
# =============================================
def func_generate_data():
    print("\n" + "=" * 60)
    print("  生成训练数据")
    print("=" * 60)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "data", "synthetic"))
    try:
        from generate_synthetic_data import (
            generate_llm_training_data,
            generate_whisper_training_data,
        )
    except ImportError:
        # 直接 import
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "data", "synthetic"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gen", os.path.join(PROJECT_ROOT, "data", "synthetic", "generate_synthetic_data.py")
        )
        gen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen)
        generate_llm_training_data = gen.generate_llm_training_data
        generate_whisper_training_data = gen.generate_whisper_training_data

    output_dir = os.path.join(PROJECT_ROOT, "data", "synthetic")
    os.makedirs(output_dir, exist_ok=True)

    print("\n  [1/3] 生成 LLM 微调数据 (2000条)...")
    llm_data = generate_llm_training_data(num_samples=2000)
    llm_path = os.path.join(output_dir, "llm_training_data.json")
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=2)
    print(f"  完成: {llm_path} ({len(llm_data)} 条)")

    print("\n  [2/3] 生成 Whisper 微调数据 (1000条音频)...")
    whisper_data = generate_whisper_training_data(
        output_dir=output_dir, num_samples=1000
    )

    print("\n  [3/3] 生成知识库扩展数据...")
    from src.rag.knowledge_base import PetKnowledgeBase, KnowledgeEntry
    kb = PetKnowledgeBase()
    kb.save_to_json(os.path.join(output_dir, "pet_knowledge_extended.json"))

    print("\n  训练数据生成完成!")
    print(f"  输出目录: {output_dir}")


# =============================================
# 2. 训练 Whisper
# =============================================
def func_train_whisper():
    print("\n" + "=" * 60)
    print("  训练 Whisper 模型")
    print("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic", "whisper_labels.json")
    if not check_file(data_path):
        print("  请先运行 [1] 生成训练数据")
        return

    from src.models.whisper_finetune import WhisperFineTuner, WhisperFineTuneConfig

    model = input("  基础模型 (默认 openai/whisper-large-v3): ") or "openai/whisper-large-v3"
    epochs = int(input("  训练轮数 (默认 30): ") or "30")

    print(f"\n  开始训练: {model}, {epochs} epochs")
    print("  (需要 GPU, 16GB+ 显存)")

    config = WhisperFineTuneConfig(
        model_name=model,
        num_train_epochs=epochs,
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints", "whisper"),
    )
    finetuner = WhisperFineTuner(config)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    eval_data = data[split:]

    save_path = finetuner.train(train_data, eval_data)
    print(f"\n  训练完成! 模型保存到: {save_path}")


# =============================================
# 3. 训练 LLM
# =============================================
def func_train_llm():
    print("\n" + "=" * 60)
    print("  训练 LLM 模型")
    print("=" * 60)

    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic", "llm_training_data.json")
    if not check_file(data_path):
        print("  请先运行 [1] 生成训练数据")
        return

    from src.models.llm_finetune import LLMFineTuner, LLMFineTuneConfig

    model = input("  基础模型 (默认 Qwen/Qwen2.5-7B-Instruct): ") or "Qwen/Qwen2.5-7B-Instruct"
    epochs = int(input("  训练轮数 (默认 10): ") or "10")

    print(f"\n  开始训练: {model}, {epochs} epochs")
    print("  (需要 GPU, 24GB+ 显存)")

    config = LLMFineTuneConfig(
        base_model=model,
        num_train_epochs=epochs,
        output_dir=os.path.join(PROJECT_ROOT, "checkpoints", "llm"),
    )
    finetuner = LLMFineTuner(config)

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    split = int(len(data) * 0.9)
    train_data = data[:split]
    eval_data = data[split:]

    save_path = finetuner.train(train_data, eval_data)
    print(f"\n  训练完成! LoRA adapter 保存到: {save_path}")


# =============================================
# 4. 宠物声音 → 人类语言
# =============================================
def func_pet_to_text():
    print("\n" + "=" * 60)
    print("  宠物声音 → 人类语言")
    print("=" * 60)

    audio_path = input("  请输入宠物声音文件路径: ").strip()
    if not check_file(audio_path):
        return

    pet_type = get_pet_type()
    breed = input("  品种 (默认 通用): ") or "通用"

    whisper_model = os.path.join(PROJECT_ROOT, "checkpoints", "whisper", "final")
    llm_adapter = os.path.join(PROJECT_ROOT, "checkpoints", "llm", "final")

    print("\n  正在加载模型...")
    from src.inference.pipeline import TranslationPipeline
    pipeline = TranslationPipeline(
        whisper_model_path=whisper_model if os.path.exists(whisper_model) else None,
        llm_adapter_path=llm_adapter if os.path.exists(llm_adapter) else None,
    )

    print("  正在分析宠物声音...\n")
    result = pipeline.pet_sound_to_text(audio_path, pet_type=pet_type, breed=breed)

    print("-" * 60)
    print(f"  Whisper 识别: {result.get('whisper_transcription', '')}")
    print(f"\n  分析结果:\n{result.get('analysis', '')}")
    print("-" * 60)


# =============================================
# 5. 人类语言 → 宠物声音
# =============================================
def func_text_to_pet():
    print("\n" + "=" * 60)
    print("  人类语言 → 宠物声音")
    print("=" * 60)

    text = input("  请输入要说的话 (如: 过来吃饭): ").strip()
    if not text:
        print("  输入不能为空")
        return

    pet_type = get_pet_type()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        ENTERTAINMENT_DIR, pet_type, f"translation_{timestamp}.wav"
    )

    print("\n  正在生成宠物声音...")
    from src.models.audio_generation import AudioGenerator
    gen = AudioGenerator()
    gen.human_speech_to_pet_sound(
        speech_text=text, target_pet=pet_type, output_path=output_path
    )
    print(f"\n  生成完成!")
    print(f"  输出文件: {output_path}")


# =============================================
# 6. 对话模式
# =============================================
def func_chat():
    print("\n" + "=" * 60)
    print("  对话模式 (输入 quit 退出)")
    print("=" * 60)

    whisper_model = os.path.join(PROJECT_ROOT, "checkpoints", "whisper", "final")
    llm_adapter = os.path.join(PROJECT_ROOT, "checkpoints", "llm", "final")

    from src.inference.pipeline import TranslationPipeline
    pipeline = TranslationPipeline(
        whisper_model_path=whisper_model if os.path.exists(whisper_model) else None,
        llm_adapter_path=llm_adapter if os.path.exists(llm_adapter) else None,
    )

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue
        response = pipeline.chat(user_input)
        print(f"\n喵汪翻译官: {response}")


# =============================================
# 7-11. 娱乐模式
# =============================================
def func_entertain(mode_name: str):
    print(f"\n{'=' * 60}")
    print(f"  逗宠模式: {mode_name}")
    print(f"{'=' * 60}")

    pet_type = get_pet_type()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        ENTERTAINMENT_DIR, pet_type, f"{mode_name}_{timestamp}.wav"
    )

    from src.agent.entertainment import EntertainmentEngine
    engine = EntertainmentEngine()

    print(f"\n  正在生成 {mode_name} 音频...")

    if mode_name == "xiangsheng":
        result = engine.play_xiangsheng(pet_type=pet_type, output_path=output_path)
        print(f"\n  标题: {result.get('title', '')}")
        print(f"\n  相声内容:")
        for i, line in enumerate(result.get("script", [])):
            speaker = "逗哏" if i % 2 == 0 else "捧哏"
            print(f"    {speaker}: {line}")
    elif mode_name == "singing":
        result = engine.play_singing(pet_type=pet_type, output_path=output_path)
        print(f"\n  歌曲名: {result.get('title', '')}")
        print(f"\n  歌词:\n{result.get('lyrics', '')}")
    elif mode_name == "soothing":
        result = engine.play_soothing(
            pet_type=pet_type, duration=60, output_path=output_path
        )
        print(f"\n  安抚音类型: {result.get('sound_type', '')}")
        print(f"  描述: {result.get('description', '')}")
    elif mode_name == "teasing":
        result = engine.play_teasing(pet_type=pet_type, output_path=output_path)
    elif mode_name == "storytelling":
        result = engine.play_storytelling(pet_type=pet_type, output_path=output_path)
        print(f"\n  故事:\n{result.get('story', '')}")

    duration = result.get("duration", 0)
    print(f"\n  输出文件: {output_path}")
    print(f"  时长: {duration:.1f} 秒")


# =============================================
# 12. 查看所有娱乐模式
# =============================================
def func_list_modes():
    print("\n" + "=" * 60)
    print("  所有娱乐模式")
    print("=" * 60)

    pet_type = get_pet_type()
    from src.agent.entertainment import EntertainmentEngine
    engine = EntertainmentEngine()
    modes = engine.list_modes(pet_type)

    for mode_id, info in modes.items():
        print(f"\n  [{mode_id}] {info['name']}")
        print(f"  描述: {info['description']}")
        if "scripts" in info:
            print(f"  相声段子: {', '.join(info['scripts'])}")
        if "songs" in info:
            print(f"  歌曲: {', '.join(info['songs'])}")
        if "sounds" in info:
            print(f"  安抚音: {', '.join(info['sounds'])}")


# =============================================
# 13. 定时逗宠
# =============================================
def func_schedule():
    print("\n" + "=" * 60)
    print("  定时逗宠调度器")
    print("=" * 60)

    pet_type = get_pet_type()
    times = int(input("  每天互动次数 (3-5, 默认 3): ") or "3")
    times = max(1, min(times, 10))

    modes_input = input(
        "  模式 (逗号分隔, 默认 xiangsheng,singing,soothing): "
    ).strip()
    modes = [m.strip() for m in modes_input.split(",")] if modes_input else None

    from src.agent.scheduler import EntertainmentScheduler, ScheduleConfig

    config = ScheduleConfig(
        pet_type=pet_type,
        times_per_day=times,
        modes=modes or ["xiangsheng", "singing", "soothing"],
        output_dir=ENTERTAINMENT_DIR,
    )
    scheduler = EntertainmentScheduler(config)
    scheduler.set_times_per_day(times)

    # 回调
    def on_complete(record):
        print(f"\n  [互动完成] 模式: {record['mode']}, 文件: {record['output_path']}")

    scheduler.add_callback(on_complete)
    scheduler.start()

    print(f"\n  调度器已启动!")
    print(f"  时间表: {', '.join(config.fixed_times)}")
    print(f"  下次运行: {scheduler.get_next_run()}")
    print(f"\n  按 Ctrl+C 停止\n")

    try:
        while True:
            time.sleep(60)
            from datetime import datetime
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"等待中... 下次: {scheduler.get_next_run()}")
    except KeyboardInterrupt:
        scheduler.stop()
        config_path = os.path.join(PROJECT_ROOT, "config", "schedule.json")
        scheduler.save_config(config_path)
        print("\n  调度器已停止,配置已保存")


# =============================================
# 14. 多智能体模式
# =============================================
def func_multi_agent():
    print("\n" + "=" * 60)
    print("  多智能体自动编排模式")
    print("=" * 60)

    pet_type = get_pet_type()
    times = int(input("  每天互动次数 (3-5, 默认 5): ") or "5")

    from src.agent.multi_agent_system import MultiAgentEntertainmentSystem

    system = MultiAgentEntertainmentSystem(
        pet_type=pet_type,
        times_per_day=times,
        output_dir=ENTERTAINMENT_DIR,
    )

    print("\n  [首次运行]")
    system.run_once()

    status = system.get_status()
    print(f"\n  {'=' * 40}")
    print(f"  系统状态")
    print(f"  {'=' * 40}")
    print(f"  宠物类型: {status['pet_type']}")
    print(f"  宠物情绪: {status['pet_state']['mood']}")
    print(f"  能量值: {status['pet_state']['energy']:.1f}")
    print(f"  今日互动: {status['pet_state']['interactions_today']} 次")
    print(f"  偏好模式: {', '.join(status['pet_state']['preferred_modes'])}")

    print(f"\n  启动定时调度 (每天 {times} 次)...")
    system.start_scheduled()
    print(f"\n  按 Ctrl+C 停止\n")

    try:
        while True:
            time.sleep(60)
            from datetime import datetime
            status = system.get_status()
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                  f"宠物: {status['pet_state']['mood']}, "
                  f"下次: {status['next_run']}")
    except KeyboardInterrupt:
        system.stop_scheduled()
        log_path = os.path.join(OUTPUT_DIR, "interaction_log.json")
        system.save_log(log_path)
        print("\n  系统已停止,日志已保存")


# =============================================
# 15. 查看系统状态
# =============================================
def func_status():
    print("\n" + "=" * 60)
    print("  系统状态")
    print("=" * 60)

    # 检查模型
    whisper_path = os.path.join(PROJECT_ROOT, "checkpoints", "whisper", "final")
    llm_path = os.path.join(PROJECT_ROOT, "checkpoints", "llm", "final")

    print(f"\n  [模型状态]")
    print(f"  Whisper 微调模型: {'已训练' if os.path.exists(whisper_path) else '未训练'}")
    if os.path.exists(whisper_path):
        print(f"    路径: {whisper_path}")
    print(f"  LLM LoRA adapter: {'已训练' if os.path.exists(llm_path) else '未训练'}")
    if os.path.exists(llm_path):
        print(f"    路径: {llm_path}")

    # 检查数据
    whisper_data = os.path.join(PROJECT_ROOT, "data", "synthetic", "whisper_labels.json")
    llm_data = os.path.join(PROJECT_ROOT, "data", "synthetic", "llm_training_data.json")

    print(f"\n  [训练数据]")
    if os.path.exists(whisper_data):
        with open(whisper_data, "r") as f:
            count = len(json.load(f))
        print(f"  Whisper 数据: {count} 条")
    else:
        print(f"  Whisper 数据: 未生成")
    if os.path.exists(llm_data):
        with open(llm_data, "r") as f:
            count = len(json.load(f))
        print(f"  LLM 数据: {count} 条")
    else:
        print(f"  LLM 数据: 未生成")

    # 检查输出
    print(f"\n  [输出文件]")
    total_files = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        wavs = [f for f in files if f.endswith(".wav")]
        total_files += len(wavs)

    print(f"  音频文件总数: {total_files}")
    for pet in ["cat", "dog"]:
        pet_dir = os.path.join(ENTERTAINMENT_DIR, pet)
        if os.path.exists(pet_dir):
            wavs = [f for f in os.listdir(pet_dir) if f.endswith(".wav")]
            print(f"  {pet} 娱乐音频: {len(wavs)} 个")

    # 配置
    config_path = os.path.join(PROJECT_ROOT, "config", "schedule.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            sched = json.load(f)
        print(f"\n  [定时配置]")
        print(f"  每日次数: {sched.get('times_per_day', 'N/A')}")
        print(f"  时间表: {', '.join(sched.get('fixed_times', []))}")

    # 互动日志
    log_path = os.path.join(OUTPUT_DIR, "interaction_log.json")
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            logs = json.load(f)
        print(f"\n  [互动日志]")
        print(f"  总互动次数: {len(logs)}")
        if logs:
            last = logs[-1]
            print(f"  最后互动: {last.get('timestamp', 'N/A')}")
            obs = last.get("observation", {})
            print(f"  最后模式: {obs.get('mode_played', 'N/A')}")
            print(f"  宠物参与度: {obs.get('engagement', 0):.0%}")


# =============================================
# 16. Whisper 优化推理
# =============================================
def func_whisper_optimized():
    print("\n" + "=" * 60)
    print("  Whisper 优化推理 (CTranslate2 加速)")
    print("=" * 60)

    audio_path = input("  请输入音频文件路径: ").strip()
    if not check_file(audio_path):
        return

    from src.models.whisper_optimized import (
        OptimizedWhisperInference,
        OptimizedWhisperConfig,
    )

    whisper_path = os.path.join(PROJECT_ROOT, "checkpoints", "whisper", "final")
    config = OptimizedWhisperConfig(
        model_path=whisper_path if os.path.exists(whisper_path) else "large-v3",
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        compute_type="int8_float16",
        vad_filter=True,
        beam_size=5,
    )

    print("\n  加载优化模型...")
    whisper = OptimizedWhisperInference(config)

    print("  推理中 (CTranslate2 加速)...")
    start = time.time()
    result = whisper.transcribe_single(audio_path)
    elapsed = time.time() - start

    print(f"\n  推理耗时: {elapsed:.2f}s")
    print(f"  识别结果: {result.get('text', '')}")
    print(f"  语言: {result.get('language', '')} "
          f"(置信度: {result.get('language_probability', 0):.2f})")
    print(f"  时长: {result.get('duration', 0):.1f}s")

    if result.get("segments"):
        print(f"\n  分段:")
        for seg in result["segments"][:5]:
            print(f"    [{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")


# =============================================
# 17. LLM vLLM 推理
# =============================================
def func_llm_vllm():
    print("\n" + "=" * 60)
    print("  LLM vLLM 推理 (AWQ 量化 + Speculative Decoding)")
    print("=" * 60)

    from src.models.llm_optimized import OptimizedLLMInference, OptimizedLLMConfig

    llm_path = os.path.join(PROJECT_ROOT, "checkpoints", "llm", "final")
    use_quant = input("  启用 AWQ 量化? (y/N): ").strip().lower() == "y"
    use_draft = input("  启用投机解码? (y/N): ").strip().lower() == "y"

    config = OptimizedLLMConfig(
        model_path=llm_path if os.path.exists(llm_path) else "Qwen/Qwen2.5-7B-Instruct",
        quantization="awq" if use_quant else None,
        draft_model="Qwen/Qwen2.5-0.5B" if use_draft else None,
        enable_prefix_caching=True,
    )

    print("\n  加载 vLLM 模型...")
    llm = OptimizedLLMInference(config)

    print("\n  进入对话模式 (输入 quit 退出)")
    print("  提示: 可描述宠物声音,LLM 会分析情绪\n")

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        if not user_input:
            continue

        start = time.time()
        response = llm.chat(
            user_input,
            system_prompt="你是一个宠物语言翻译专家。",
            max_tokens=256,
        )
        elapsed = time.time() - start
        print(f"\nAI: {response}")
        print(f"  (耗时: {elapsed:.2f}s)\n")


# =============================================
# 18. RAG 混合检索测试
# =============================================
def func_rag_test():
    print("\n" + "=" * 60)
    print("  RAG 混合检索测试")
    print("=" * 60)

    from src.rag.knowledge_base import PetKnowledgeBase
    from src.rag.optimized_retriever import HybridRetriever, OptimizedRAGConfig

    print("  构建知识库索引...")
    kb = PetKnowledgeBase()
    retriever = HybridRetriever(OptimizedRAGConfig())
    retriever.build_index(kb)

    print("\n  进入检索模式 (输入 quit 退出)\n")

    while True:
        query = input("查询: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            continue

        print(f"\n  查询: {query}")
        results = retriever.search(query, top_k=3)

        print(f"  检索结果 ({len(results)} 条):")
        for i, r in enumerate(results, 1):
            score = r.get("rerank_score", r.get("score", 0))
            text = r.get("text", "")[:100]
            print(f"\n  [{i}] 相关度: {score:.3f}")
            print(f"      {text}...")

        context = retriever.get_context_for_llm(query)
        print(f"\n  LLM 上下文:\n{context[:500]}...\n")


# =============================================
# 主函数
# =============================================
def main():
    # 清屏
    os.system("clear" if os.name != "nt" else "cls")
    print_banner()

    while True:
        print_menu()
        choice = input("请选择功能 [0-18]: ").strip()

        if choice == "0":
            print("\n  再见!\n")
            break
        elif choice == "1":
            func_generate_data()
        elif choice == "2":
            func_train_whisper()
        elif choice == "3":
            func_train_llm()
        elif choice == "4":
            func_pet_to_text()
        elif choice == "5":
            func_text_to_pet()
        elif choice == "6":
            func_chat()
        elif choice == "7":
            func_entertain("xiangsheng")
        elif choice == "8":
            func_entertain("singing")
        elif choice == "9":
            func_entertain("soothing")
        elif choice == "10":
            func_entertain("teasing")
        elif choice == "11":
            func_entertain("storytelling")
        elif choice == "12":
            func_list_modes()
        elif choice == "13":
            func_schedule()
        elif choice == "14":
            func_multi_agent()
        elif choice == "15":
            func_status()
        elif choice == "16":
            func_whisper_optimized()
        elif choice == "17":
            func_llm_vllm()
        elif choice == "18":
            func_rag_test()
        else:
            print("  无效选择,请重新输入")

        if choice != "0":
            input("\n  按回车键继续...")
            os.system("clear" if os.name != "nt" else "cls")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  已退出\n")
    except Exception as e:
        print(f"\n  错误: {e}")
        import traceback
        traceback.print_exc()
