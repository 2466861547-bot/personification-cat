"""
训练宠物声音分类器
使用 data/raw/cat_sounds 和 data/raw/dog_sounds 中的真实数据
自动扫描目录结构，按文件名中的情绪标签分类

用法:
  python scripts/train_pet_classifier.py
  python scripts/train_pet_classifier.py --n-estimators 200
"""

import os
import sys
import json
import argparse
from collections import defaultdict

# 添加项目根到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.pet_sound_classifier import PetSoundClassifier, EMOTION_LABELS


def infer_emotion_from_filename(filename: str) -> str:
    """从文件名推断情绪标签"""
    basename = os.path.splitext(filename)[0].lower()
    
    # 直接匹配情绪标签
    for label in EMOTION_LABELS:
        if label in basename:
            return label
    
    # 常见关键词映射
    keyword_map = {
        "hungry": ["hungry", "feeding", "food", "eat", "饿", "吃"],
        "happy": ["happy", "joy", "playful", "play", "开心", "玩"],
        "angry": ["angry", "aggressive", "snarl", "bark_angry", "怒", "凶"],
        "fearful": ["fear", "scared", "afraid", "害怕", "恐"],
        "seek_attention": ["attention", "seek", "call", "voice", "attention"],
        "content": ["content", "purr", "relaxed", "satisfied", "满足", "咕噜"],
        "pain": ["pain", "hurt", "injury", "sick", "痛", "伤"],
        "curious": ["curious", "interest", "sniff", "好奇"],
        "lonely": ["lonely", "alone", "cry", "whine", "孤独", "呜咽"],
        "anxious": ["anxious", "stress", "panic", "焦虑"],
        "excited": ["excited", "aroused", "energetic", "兴奋"],
        "frustrated": ["frustrated", "annoyed", "irritated", "挫败"],
        "relaxed": ["relaxed", "calm", "peaceful", "平静"],
        "alert": ["alert", "alerted", "warning", "警戒"],
        "territorial": ["territory", "territorial", "marking", "领地"],
        "greeting": ["greet", "hello", "welcome", "问候", "打招呼"],
        "confused": ["confuse", "confused", "puzzle", "困惑"],
        "jealous": ["jealous", "envy", "吃醋"],
        "sad": ["sad", "depressed", "down", "悲伤"],
        "playful": ["playful", "play", "toy", "ball", "玩耍"],
    }
    
    for label, keywords in keyword_map.items():
        for kw in keywords:
            if kw in basename:
                return label
    
    # 未匹配到的默认 fallback
    return "seek_attention"


def scan_audio_directory(directory: str) -> dict:
    """扫描音频目录，按情绪分类"""
    emotion_files = defaultdict(list)
    
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith((".wav", ".mp3", ".flac", ".ogg")):
                emotion = infer_emotion_from_filename(f)
                emotion_files[emotion].append(os.path.join(root, f))
    
    return dict(emotion_files)


def main():
    parser = argparse.ArgumentParser(description="训练宠物声音分类器")
    parser.add_argument("--cat-dir", default="./data/raw/cat_sounds", help="猫声音目录")
    parser.add_argument("--dog-dir", default="./data/raw/dog_sounds", help="狗声音目录")
    parser.add_argument("--output", default="./checkpoints/pet_classifier", help="模型输出目录")
    parser.add_argument("--n-estimators", type=int, default=200, help="随机森林树数量")
    parser.add_argument("--auto-generate", action="store_true",
                        help="如果数据不足，自动生成合成数据")
    args = parser.parse_args()

    print("=" * 60)
    print("🐾 宠物声音分类器训练")
    print("=" * 60)

    # 1. 扫描数据
    print(f"\n📂 扫描音频数据...")
    cat_emotions = scan_audio_directory(args.cat_dir) if os.path.isdir(args.cat_dir) else {}
    dog_emotions = scan_audio_directory(args.dog_dir) if os.path.isdir(args.dog_dir) else {}

    # 合并猫和狗的数据，都用于训练分类器 (通用情绪识别)
    all_emotions = defaultdict(list)
    for emotion, files in cat_emotions.items():
        all_emotions[emotion].extend(files)
    for emotion, files in dog_emotions.items():
        all_emotions[emotion].extend(files)

    total_files = sum(len(f) for f in all_emotions.values())
    print(f"  猫声音: {sum(len(f) for f in cat_emotions.values())} 个")
    print(f"  狗声音: {sum(len(f) for f in dog_emotions.values())} 个")
    print(f"  总计: {total_files} 个音频文件")

    for emotion, files in sorted(all_emotions.items()):
        print(f"    [{emotion}]: {len(files)} 个样本")

    if total_files == 0:
        print("\n⚠️  没有找到任何音频文件！")
        print("请将音频文件放入 data/raw/cat_sounds/ 和 data/raw/dog_sounds/")
        print("文件名应包含情绪关键词，如 'hungry_feeding_time.wav'")
        return

    # 2. 检查数据量，不足时提示
    if total_files < 5:
        print("\n⚠️  数据量非常少 (<5 个样本)，分类器效果会很有限")
        print("建议生成更多测试数据:")
        print("  python scripts/generate_test_audio.py")

    # 3. 构造训练数据目录结构 (为了兼容 PetSoundClassifier.train 方法)
    import tempfile
    import shutil

    # 创建临时目录结构
    temp_dir = os.path.join(tempfile.gettempdir(), "pet_clf_train_data")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    audio_dirs = {}
    for emotion, files in all_emotions.items():
        emotion_dir = os.path.join(temp_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        for src_file in files:
            dst_file = os.path.join(emotion_dir, os.path.basename(src_file))
            if src_file != dst_file:
                shutil.copy2(src_file, dst_file)
        audio_dirs[emotion] = emotion_dir

    # 4. 训练
    classifier = PetSoundClassifier()
    classifier.train(
        audio_dirs=audio_dirs,
        model_dir=args.output,
        n_estimators=args.n_estimators,
    )

    # 5. 快速验证
    print("\n" + "=" * 60)
    print("🔍 快速验证分类器")
    print("=" * 60)

    test_files = []
    for files_list in all_emotions.values():
        test_files.extend(files_list[:2])  # 每个情绪取前2个测试
        if len(test_files) >= 10:
            break

    if test_files and classifier.is_ready:
        correct = 0
        total = 0
        for f in test_files:
            true_emotion = infer_emotion_from_filename(os.path.basename(f))
            result = classifier.predict(f)
            predicted = result["emotion"]
            match = "✅" if predicted == true_emotion else "❌"
            print(f"  {match} {os.path.basename(f):40s} 预期={true_emotion:15s} 预测={predicted:15s} 置信度={result['confidence']:.2f}")
            if predicted == true_emotion:
                correct += 1
            total += 1

        if total > 0:
            print(f"\n  Top1 准确率: {correct}/{total} = {correct/total*100:.1f}%")

    # 清理临时文件
    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n✅ 宠物声音分类器训练完成！")
    print(f"   模型路径: {os.path.abspath(args.output)}")
    print(f"   现在可以使用: python scripts/inference.py --mode pet_to_text ...")


if __name__ == "__main__":
    main()