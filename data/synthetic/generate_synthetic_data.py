"""
私域训练数据合成: 生成宠物声音-文本配对训练数据
包括:
  1. Whisper 微调数据 (音频→文字)
  2. LLM 微调数据 (文字→情绪解读)
  3. 知识库补充数据
"""

import os
import json
import random
import numpy as np
import librosa
from typing import List, Dict
from scipy.io import wavfile


# =====================================
# 1. 声音描述模板库
# =====================================

CAT_SOUND_TEMPLATES = {
    "hungry": [
        "短促的喵叫，重复2-3次，音调上扬",
        "高频持续的喵喵叫，伴随蹭腿",
        "急促的喵叫，看向食物方向",
    ],
    "happy": [
        "轻柔的短喵，伴随呼噜声",
        "柔和的喵叫，尾巴竖起",
        "愉悦的喵叫，蹭主人手",
    ],
    "angry": [
        "低沉的嘶嘶声，弓背炸毛",
        "连续的低吼，耳朵后压",
        "尖锐的嘶叫，瞳孔放大",
    ],
    "fear": [
        "颤抖的哀鸣，躲在角落",
        "尖锐的长嚎，身体缩成一团",
        "微弱的呜咽，不敢出来",
    ],
    "seek_attention": [
        "持续有节奏的喵叫，跟随主人",
        "用爪子扒门，持续喵叫",
        "坐在主人面前，反复喵叫",
    ],
    "content": [
        "低频持续的呼噜声，闭眼",
        "满足的呼噜，趴在温暖处",
        "轻柔的呼噜，缓慢眨眼",
    ],
    "pain": [
        "突然的尖叫，异常尖锐",
        "持续的哀嚎，不愿被触碰",
        "不规律的呻吟，蜷缩不动",
    ],
    "alert": [
        "短促高频的咯咯声，盯着窗外",
        "尖锐的喵叫，耳朵转向声源",
        "急促的叫声，身体前倾",
    ],
}

DOG_SOUND_TEMPLATES = {
    "hungry": [
        "短促的汪汪叫2声，舔食盆",
        "急促的汪叫，在食物区徘徊",
        "低沉的呜咽，看向主人",
    ],
    "happy": [
        "轻快的汪汪叫，摇尾巴",
        "活泼的叫声，前肢下压",
        "欢快的汪叫，转圈",
    ],
    "angry": [
        "低沉的咆哮，露牙",
        "连续的低吼，毛发竖起",
        "嘶哑的低吼，身体僵硬",
    ],
    "fear": [
        "颤抖的呜咽，夹着尾巴",
        "尖锐的哀嚎，躲在沙发后",
        "持续的呜呜叫，耳朵下垂",
    ],
    "alert": [
        "连续快速的汪汪叫",
        "急促的吠叫，朝门口方向",
        "短促有力的汪叫，竖耳",
    ],
    "playful": [
        "活泼的汪汪叫，前肢下压",
        "兴奋的叫声，叼着玩具",
        "欢快的吠叫，蹦跳",
    ],
    "sad": [
        "低沉的呜咽，趴在地上",
        "缓慢的哀嚎，无精打采",
        "持续的呜呜叫，不看主人",
    ],
    "pain": [
        "突然的尖叫，跛行",
        "异常的低吟，不愿移动",
        "尖锐的哀嚎，躲避触碰",
    ],
}


# =====================================
# 2. LLM 训练数据模板
# =====================================

def generate_llm_training_data(num_samples: int = 2000) -> List[Dict]:
    """生成 LLM 微调训练数据"""
    training_data = []

    pet_types = ["cat", "dog"]
    # 16 猫 + 16 狗，必须含矮脚拿破仑
    breeds = {
        "cat": [
            "英短", "美短", "布偶", "橘猫", "暹罗", "波斯", "缅因", "狸花",
            "矮脚拿破仑", "苏格兰折耳", "阿比西尼亚", "孟加拉豹猫",
            "斯芬克斯无毛猫", "俄罗斯蓝猫", "加菲猫", "美国卷耳猫",
        ],
        "dog": [
            "金毛", "拉布拉多", "哈士奇", "柯基", "泰迪", "边牧", "德牧", "柴犬",
            "萨摩耶", "阿拉斯加", "比熊", "博美", "法国斗牛犬",
            "雪纳瑞", "比格犬", "秋田犬",
        ],
    }
    emotions = list(CAT_SOUND_TEMPLATES.keys())
    contexts = [
        "早上刚起床", "午后休息时间", "主人刚回家", "准备吃饭时",
        "看到窗外有小鸟", "听到门铃声", "陌生人在附近", "和其他宠物在一起",
        "独自在家", "天气不好时", "去医院路上", "玩耍后",
    ]

    for i in range(num_samples):
        pet_type = random.choice(pet_types)
        breed = random.choice(breeds[pet_type])
        emotion = random.choice(emotions)
        context = random.choice(contexts)

        templates = CAT_SOUND_TEMPLATES if pet_type == "cat" else DOG_SOUND_TEMPLATES
        sound_desc = random.choice(templates.get(emotion, ["普通叫声"]))

        # 生成指令微调样本
        sample = {
            "system": "你是一个宠物语言翻译专家，能够根据宠物声音特征和情境判断其情绪和需求，并给出专业建议。",
            "input": (
                f"宠物类型: {pet_type}\n"
                f"品种: {breed}\n"
                f"声音特征: {sound_desc}\n"
                f"情境: {context}\n"
                f"请分析这只{breed}({pet_type})的情绪状态和需求。"
            ),
            "output": generate_analysis_output(pet_type, breed, emotion, sound_desc, context),
        }
        training_data.append(sample)

    return training_data


def generate_analysis_output(
    pet_type: str,
    breed: str,
    emotion: str,
    sound_desc: str,
    context: str,
) -> str:
    """生成分析输出"""
    emotion_cn = {
        "hungry": "饥饿", "happy": "开心", "angry": "愤怒", "fear": "恐惧",
        "seek_attention": "求关注", "content": "满足", "pain": "疼痛",
        "alert": "警觉", "playful": "想玩耍", "sad": "悲伤",
    }

    advice_map = {
        "hungry": f"建议检查上次喂食时间，适量补充食物，{breed}的每日喂食量应按体重计算。",
        "happy": f"{breed}心情愉悦，可以适度互动抚摸，继续维持良好的互动关系。",
        "angry": f"请立即停止当前行为，给{breed}足够的空间，避免直视，等其平静后再接近。",
        "fear": f"为{breed}提供安全的躲藏空间，保持环境安静，不要强迫互动。",
        "seek_attention": f"回应{breed}的需求，进行5-10分钟的互动(抚摸/玩耍)。",
        "content": f"{breed}状态良好，保持当前环境和互动节奏。",
        "pain": f"请立即检查{breed}身体状况，联系兽医进行专业诊断。",
        "alert": f"观察{breed}警觉的方向，确认安全后安抚其情绪。",
        "playful": f"与{breed}互动玩耍10-15分钟，满足其运动需求。",
        "sad": f"增加对{breed}的陪伴时间，可提供益智玩具缓解情绪。",
    }

    return (
        f"## 情绪解读\n"
        f"这只{breed}当前的情绪状态是「{emotion_cn.get(emotion, emotion)}」。\n\n"
        f"## 声音分析\n"
        f"声音特征: {sound_desc}\n"
        f"这种叫声通常表明宠物{emotion_cn.get(emotion, emotion)}的状态。\n\n"
        f"## 情境分析\n"
        f"当前情境: {context}\n"
        f"在{context}的情况下，{breed}出现此类声音属于{'正常' if emotion in ['happy', 'content', 'seek_attention'] else '需要关注'}的表现。\n\n"
        f"## 建议措施\n"
        f"{advice_map.get(emotion, '请持续观察宠物状态。')}\n"
    )


# =====================================
# 3. Whisper 训练数据生成 (合成音频)
# =====================================

def generate_synthetic_audio(
    pet_type: str,
    emotion: str,
    duration: float = 2.0,
    sr: int = 16000,
) -> np.ndarray:
    """生成合成宠物声音(用于 Whisper 微调)"""
    t = np.linspace(0, duration, int(sr * duration))
    templates = CAT_SOUND_TEMPLATES if pet_type == "cat" else DOG_SOUND_TEMPLATES

    if pet_type == "cat":
        if emotion == "content":
            # 呼噜声 25Hz
            audio = 0.3 * np.sin(2 * np.pi * 25 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
        elif emotion == "hungry":
            # 高频喵叫
            freq = 800 + 200 * np.sin(2 * np.pi * 3 * t)
            audio = 0.4 * np.sin(2 * np.pi * freq * t) * (np.mod(t * 2, 1) < 0.4).astype(float)
        elif emotion == "angry":
            # 嘶嘶声(噪声)
            audio = 0.4 * np.random.randn(len(t)) * (np.mod(t * 3, 1) < 0.3).astype(float)
        elif emotion == "alert":
            # 咯咯声(高频短促)
            audio = np.zeros_like(t)
            for start in np.arange(0, duration, 0.15):
                mask = (t >= start) & (t < start + 0.05)
                audio[mask] = 0.3 * np.sin(2 * np.pi * 1500 * t[mask])
        else:
            # 普通喵叫
            freq = 600 + 100 * np.sin(2 * np.pi * 4 * t)
            audio = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
    else:
        # 狗
        if emotion == "happy":
            audio = np.zeros_like(t)
            for bark_time in [0.1, 0.5, 0.9]:
                mask = (t > bark_time) & (t < bark_time + 0.15)
                audio[mask] = 0.5 * np.sin(2 * np.pi * 500 * t[mask])
        elif emotion == "angry":
            audio = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))
        elif emotion == "alert":
            audio = np.zeros_like(t)
            for bark_time in np.arange(0.1, duration, 0.2):
                mask = (t > bark_time) & (t < bark_time + 0.1)
                audio[mask] = 0.4 * np.sin(2 * np.pi * 600 * t[mask])
        elif emotion == "playful":
            audio = np.zeros_like(t)
            for bark_time in [0.1, 0.3, 0.5, 0.7, 0.9]:
                mask = (t > bark_time) & (t < bark_time + 0.08)
                audio[mask] = 0.4 * np.sin(2 * np.pi * 700 * t[mask])
        else:
            audio = 0.3 * np.sin(2 * np.pi * 400 * t) * np.exp(-t * 0.5)

    # 添加背景噪声
    noise = 0.01 * np.random.randn(len(audio))
    audio = audio + noise

    # 标准化
    max_val = np.max(np.abs(audio)) + 1e-10
    audio = audio / max_val * 0.9

    return audio.astype(np.float32)


def generate_whisper_training_data(
    output_dir: str = "./data/synthetic",
    num_samples: int = 1000,
) -> List[Dict]:
    """生成 Whisper 微调训练数据(音频+标签)"""
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    training_data = []
    pet_types = ["cat", "dog"]
    emotions = list(CAT_SOUND_TEMPLATES.keys())

    for i in range(num_samples):
        pet_type = random.choice(pet_types)
        emotion = random.choice(emotions)
        templates = CAT_SOUND_TEMPLATES if pet_type == "cat" else DOG_SOUND_TEMPLATES
        text_label = random.choice(templates.get(emotion, ["普通叫声"]))

        # 生成音频
        audio = generate_synthetic_audio(pet_type, emotion, duration=random.uniform(1.5, 3.0))

        # 保存音频
        audio_path = os.path.join(audio_dir, f"sample_{i:05d}.wav")
        wavfile.write(audio_path, 16000, audio)

        # Whisper 标签格式: "[emotion] 声音描述"
        label = f"[{emotion}] {text_label}"

        training_data.append({
            "audio_path": audio_path,
            "text": label,
            "pet_type": pet_type,
            "emotion": emotion,
        })

    # 保存标签
    label_path = os.path.join(output_dir, "whisper_labels.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)

    print(f"Whisper 训练数据已生成: {len(training_data)} 条, 保存到 {output_dir}")
    return training_data


# =====================================
# 4. 主函数
# =====================================

def main():
    """生成全部训练数据"""
    output_dir = "./data/synthetic"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成 LLM 训练数据
    print("=" * 50)
    print("1. 生成 LLM 微调训练数据...")
    llm_data = generate_llm_training_data(num_samples=2000)
    llm_path = os.path.join(output_dir, "llm_training_data.json")
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_data, f, ensure_ascii=False, indent=2)
    print(f"   LLM 训练数据: {len(llm_data)} 条 → {llm_path}")

    # 2. 生成 Whisper 训练数据(含合成音频)
    print("=" * 50)
    print("2. 生成 Whisper 微调训练数据(含合成音频)...")
    whisper_data = generate_whisper_training_data(
        output_dir=output_dir,
        num_samples=1000,
    )

    # 3. 生成知识库补充数据
    print("=" * 50)
    print("3. 生成知识库补充数据...")
    from src.rag.knowledge_base import PetKnowledgeBase, KnowledgeEntry

    kb = PetKnowledgeBase()
    # 添加品种特定知识 (覆盖 16 猫 + 16 狗)
    breed_specific = [
        # === 猫 ===
        KnowledgeEntry("cat", "英短", "hungry",
            "英短特有的低沉短喵，声音比其他品种更浑厚",
            "英短猫饿了，但由于性格沉稳，叫声较含蓄",
            "英短通常不太爱叫，如果叫了说明确实饿了",
            "英短易胖，注意控制食量，建议定时定量喂食"),
        KnowledgeEntry("cat", "矮脚拿破仑", "seek_attention",
            "矮脚拿破仑的高频轻柔颤音(trill)，F0 450-750Hz",
            "矮脚拿破仑想引起注意，性格温顺亲人",
            "主人回家或长时间未互动时",
            "矮脚拿破仑腿短跳跃力弱，建议用逗猫棒在地面附近互动"),
        KnowledgeEntry("cat", "矮脚拿破仑", "content",
            "矮脚拿破仑的呼噜声 22-30Hz，持续低频",
            "矮脚拿破仑感到满足和放松，呼噜声带波斯血统的柔和",
            "被抚摸或晒太阳时",
            "保持环境安静，享受与它的亲密时光"),
        KnowledgeEntry("cat", "暹罗", "seek_attention",
            "暹罗猫特有的大声长喵，F0 520-850Hz，持续 0.4-1.2s",
            "暹罗猫被称为话痨猫，想表达需求时叫声特别响亮",
            "清晨、傍晚或想出门时",
            "暹罗猫需要大量互动，每天至少陪伴 30 分钟"),
        KnowledgeEntry("cat", "缅因", "alert",
            "缅因猫特有的啁啾颤音(chirp/trill)，F0 220-420Hz",
            "缅因猫发现窗外猎物时发出兴奋的颤音",
            "看到窗外鸟类或昆虫时",
            "可提供逗猫棒等玩具替代真实猎物"),
        KnowledgeEntry("cat", "加菲猫", "hungry",
            "加菲猫鼻音重的低沉喵叫，因短鼻鼻腔共鸣特殊",
            "加菲猫饿了，叫声带鼻音",
            "饭点前后",
            "加菲猫易泪痕，注意清洁眼部；定时定量喂食"),
        KnowledgeEntry("cat", "孟加拉豹猫", "alert",
            "孟加拉豹猫野性多变的啁啾音，F0 400-720Hz",
            "孟加拉豹猫保持野性本能，发现猎物时发出多变声音",
            "看到移动物体时",
            "提供丰富玩具满足其高运动需求"),
        KnowledgeEntry("cat", "苏格兰折耳", "content",
            "苏格兰折耳轻柔中等喵叫，呼噜声 22-28Hz",
            "苏格兰折耳感到舒适放松",
            "被抚摸时",
            "注意折耳猫骨骼遗传病，定期体检"),
        KnowledgeEntry("cat", "斯芬克斯无毛猫", "seek_attention",
            "斯芬克斯叫声大而频繁，F0 420-700Hz",
            "斯芬克斯性格外向，主动求关注",
            "主人回家时",
            "无毛猫怕冷，注意保暖"),
        # === 狗 ===
        KnowledgeEntry("dog", "哈士奇", "alert",
            "哈士奇特有的嚎叫(howl)，长而持续，F0 180-520Hz",
            "哈士奇在警觉或想表达需求时会嚎叫而非吠叫",
            "听到警笛声或想出门时",
            "哈士奇运动量大，需要每天 1-2 小时运动"),
        KnowledgeEntry("dog", "阿拉斯加", "alert",
            "阿拉斯加低沉长嚎，F0 150-380Hz，胸腔共鸣强",
            "阿拉斯加体型大，嚎叫低沉深远",
            "听到警报声或独处时",
            "阿拉斯加耐寒怕热，避免高温时段运动"),
        KnowledgeEntry("dog", "柯基", "alert",
            "柯基尖锐高频吠叫，F0 420-720Hz",
            "柯基腿短共鸣腔小，F0 偏高，警觉时频繁吠叫",
            "听到门铃或异常声响",
            "柯基易胖且腰椎脆弱，避免上下楼梯和过量喂食"),
        KnowledgeEntry("dog", "泰迪", "seek_attention",
            "泰迪高频尖锐吠叫，F0 450-780Hz，频繁",
            "泰迪体型小，爱叫求关注",
            "主人回家或看到其他动物时",
            "泰迪需定期美容，避免泪痕"),
        KnowledgeEntry("dog", "博美", "alert",
            "博美极高频尖锐吠叫，F0 500-850Hz",
            "博美超小型犬，F0 最高，警觉时吠叫频繁",
            "听到门外声响",
            "博美毛发蓬松需勤梳理，警惕气管塌陷"),
        KnowledgeEntry("dog", "法国斗牛犬", "happy",
            "法斗鼻音重的低沉吠叫，F0 220-420Hz",
            "法斗短鼻犬鼻腔共鸣特殊，声音沉闷",
            "玩耍或看到主人时",
            "法斗短鼻怕热，避免剧烈运动和高环境温度"),
        KnowledgeEntry("dog", "柴犬", "angry",
            "柴犬安静少吠但被惹时发出尖锐的柴犬尖叫",
            "柴犬性格独立，平时安静，但情绪激动时尖叫",
            "被强迫做不喜欢的事时",
            "柴犬性格倔强，避免强迫互动"),
        KnowledgeEntry("dog", "比格犬", "alert",
            "比格犬独特的 bay 长嚎吠叫，F0 300-520Hz",
            "比格犬嗅觉猎犬，发现气味或猎物时发出 bay 长嚎",
            "户外散步发现气味时",
            "比格犬精力旺盛，需大量户外运动"),
        KnowledgeEntry("dog", "秋田犬", "alert",
            "秋田犬低沉短促吠叫，F0 180-380Hz",
            "秋田犬大型犬安静，吠声低沉",
            "陌生人接近时",
            "秋田犬领地意识强，需早期社会化训练"),
        KnowledgeEntry("dog", "萨摩耶", "happy",
            "萨摩耶爱嚎叫和‘说话’，声音柔和带颤音",
            "萨摩耶性格活泼，会用嚎叫表达喜悦",
            "主人回家时",
            "萨摩耶毛发浓密，需勤梳理"),
    ]
    kb.add_entries(breed_specific)
    kb.save_to_json(os.path.join(output_dir, "pet_knowledge_extended.json"))

    print("=" * 50)
    print("全部训练数据生成完成!")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
