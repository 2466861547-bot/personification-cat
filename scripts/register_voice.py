"""注册预置声线 - 直接加载 audio_generation 模块 (绕过 __init__.py 导入链)"""
import os
import sys
import importlib.util

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 直接加载 audio_generation.py 模块，避免触发 src/models/__init__.py 的完整导入链
audio_gen_path = os.path.join(BASE_DIR, "src", "models", "audio_generation.py")
spec = importlib.util.spec_from_file_location("audio_generation", audio_gen_path)
audio_gen_module = importlib.util.module_from_spec(spec)
sys.modules["audio_generation"] = audio_gen_module
spec.loader.exec_module(audio_gen_module)

AudioGenerator = audio_gen_module.AudioGenerator

# 注册林志玲声线
ref_path = os.path.join(BASE_DIR, "data", "voices", "linzhiling_ref.wav")
AudioGenerator.register_preset_voice("林志玲", ref_path)

# 验证
print("\n✅ 注册完成！")
print(f"\n当前预置声线 ({len(AudioGenerator.PRESET_VOICE_REFS)} 个):")
for name, path in AudioGenerator.PRESET_VOICE_REFS.items():
    exists = "✅" if os.path.isfile(path) else "❌"
    print(f"  {exists} {name} → {path}")

print("\n" + "=" * 50)
print("使用方式:")
print("  python scripts/inference.py --mode pet_to_human_voice \\")
print("    --audio ./data/raw/cat_sounds/test.wav \\")
print("    --target-voice '林志玲' \\")
print("    --output ./output/linzhiling_cat.wav")
print("=" * 50)