"""
数据集划分模块: 基于真实动物学数据生成 train/val/test 三个集合
划分策略: 分层抽样 (按品种 + 情绪分层)
比例: train 80% / val 10% / test 10%

输入:
  - data/crawler/cat_breeds_real.json (16 猫真实数据)
  - data/crawler/dog_breeds_real.json (16 狗真实数据)
  - data/crawler/ethology_research.json (声学行为学研究)

输出:
  - data/dataset/train.json   (~2400 条)
  - data/dataset/val.json     (~300 条)
  - data/dataset/test.json    (~300 条)
"""

import os
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict
from copy import deepcopy

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CRAWLER_DIR = os.path.join(DATA_DIR, "crawler")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")

# 划分比例
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# 固定随机种子确保可复现
RANDOM_SEED = 42

# 16 种猫 + 16 种狗 (与 breed_profiles.py 对齐)
CAT_BREEDS = [
    "英短", "美短", "布偶", "橘猫", "暹罗", "波斯", "缅因", "狸花",
    "矮脚拿破仑", "苏格兰折耳", "阿比西尼亚", "孟加拉豹猫",
    "斯芬克斯无毛猫", "俄罗斯蓝猫", "加菲猫", "美国卷耳猫",
]
DOG_BREEDS = [
    "金毛", "拉布拉多", "哈士奇", "柯基", "泰迪", "边牧", "德牧", "柴犬",
    "萨摩耶", "阿拉斯加", "比熊", "博美", "法国斗牛犬",
    "雪纳瑞", "比格犬", "秋田犬",
]

# 情绪标签 (10 种)
EMOTIONS = [
    "hungry", "happy", "angry", "fear", "seek_attention",
    "content", "pain", "alert", "playful", "sad",
]

# 情境上下文
CONTEXTS = [
    "清晨刚起床", "午后休息时间", "主人下班回家", "准备吃饭时",
    "看到窗外有小鸟", "听到门铃声", "陌生人在附近", "和其他宠物在一起",
    "独自在家", "雷雨天气", "去医院路上", "玩耍后",
    "深夜", "吃饭中", "陌生人靠近", "听到警笛声",
]


# =====================================
# 加载真实动物学数据
# =====================================

# 品种简称 -> 全称映射 (与 crawler 输出的 breed_cn 对齐)
BREED_NAME_ALIASES = {
    # 猫
    "英短": "英国短毛猫", "美短": "美国短毛猫", "布偶": "布偶猫",
    "暹罗": "暹罗猫", "波斯": "波斯猫", "缅因": "缅因猫",
    "狸花": "狸花猫", "矮脚拿破仑": "矮脚拿破仑猫",
    "苏格兰折耳": "苏格兰折耳猫", "孟加拉豹猫": "孟加拉豹猫",
    "斯芬克斯无毛猫": "斯芬克斯无毛猫", "俄罗斯蓝猫": "俄罗斯蓝猫",
    "加菲猫": "异国短毛猫", "美国卷耳猫": "美国卷耳猫",
    "阿比西尼亚": "阿比西尼亚猫", "橘猫": "橘猫",
    # 狗
    "金毛": "金毛寻回犬", "拉布拉多": "拉布拉多寻回犬",
    "哈士奇": "西伯利亚雪橇犬", "柯基": "彭布罗克威尔士柯基",
    "泰迪": "贵宾犬（含泰迪玩具型）", "边牧": "边境牧羊犬",
    "德牧": "德国牧羊犬", "柴犬": "柴犬", "萨摩耶": "萨摩耶",
    "阿拉斯加": "阿拉斯加雪橇犬", "比熊": "比熊犬",
    "博美": "博美犬", "法国斗牛犬": "法国斗牛犬",
    "雪纳瑞": "迷你雪纳瑞", "比格犬": "比格犬", "秋田犬": "秋田犬",
}


def _resolve_breed_name(short_name: str) -> str:
    """将简称解析为爬取数据中使用的全称"""
    return BREED_NAME_ALIASES.get(short_name, short_name)


def load_real_breed_data() -> Tuple[Dict, Dict]:
    """加载爬取的真实品种数据, 按 breed_cn 索引"""
    cats = {}
    dogs = {}

    cat_path = os.path.join(CRAWLER_DIR, "cat_breeds_real.json")
    dog_path = os.path.join(CRAWLER_DIR, "dog_breeds_real.json")

    if os.path.exists(cat_path):
        with open(cat_path, "r", encoding="utf-8") as f:
            for breed in json.load(f):
                cats[breed["breed_cn"]] = breed

    if os.path.exists(dog_path):
        with open(dog_path, "r", encoding="utf-8") as f:
            for breed in json.load(f):
                dogs[breed["breed_cn"]] = breed

    return cats, dogs


def load_ethology_data() -> Dict:
    """加载声学行为学研究数据"""
    path = os.path.join(CRAWLER_DIR, "ethology_research.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# =====================================
# 样本生成 (基于真实数据)
# =====================================

def generate_sample(
    pet_type: str,
    breed_cn: str,
    emotion: str,
    context: str,
    breed_data: Dict,
    ethology: Dict,
) -> Dict:
    """基于真实品种数据生成一个训练样本"""

    # 真实声学参数
    f0_range = breed_data.get("f0_range_hz", "400-700")
    typical_sounds = breed_data.get("typical_sounds", ["普通叫声"])
    temperament = breed_data.get("temperament", "未知")
    weight = breed_data.get("weight_kg", "未知")
    lifespan = breed_data.get("lifespan_years", "未知")
    diseases = breed_data.get("genetic_diseases", [])
    ethology_notes = breed_data.get("ethology_notes", "")
    acoustic = breed_data.get("acoustic_features", {})

    # 从行为学研究获取该情绪对应的声学描述
    pet_acoustics = ethology.get("cat_acoustics" if pet_type == "cat" else "dog_acoustics", {})
    vocal_list = pet_acoustics.get("vocalizations", [])
    emotion_vocal_map = _map_emotion_to_vocalization(pet_type, emotion, vocal_list)

    # 构造声音描述 (融合真实数据)
    sound_desc = _build_sound_description(
        pet_type, breed_cn, emotion, typical_sounds, emotion_vocal_map
    )

    # 构造 LLM 输出
    analysis_output = _build_analysis_output(
        pet_type, breed_cn, emotion, sound_desc, context,
        f0_range, temperament, weight, lifespan, diseases, ethology_notes,
        emotion_vocal_map
    )

    # 真实声学特征向量 (用于 Whisper 训练)
    f0_low, f0_high = _parse_f0_range(f0_range)
    f0_mean = (f0_low + f0_high) / 2
    duration_range = emotion_vocal_map.get("typical_duration_s", [0.3, 1.5])
    centroid_range = emotion_vocal_map.get("spectral_centroid_hz", [1500, 3000])
    voiced_range = emotion_vocal_map.get("voiced_ratio", [0.7, 0.95])

    return {
        "id": f"{pet_type}_{breed_cn}_{emotion}_{context[:2]}_{random.randint(1000, 9999)}",
        "system": "你是一个宠物语言翻译专家，基于动物行为学和声学研究根据宠物声音特征和情境判断其情绪和需求，并给出专业建议。",
        "input": (
            f"宠物类型: {pet_type}\n"
            f"品种: {breed_cn}\n"
            f"声音特征: {sound_desc}\n"
            f"情境: {context}\n"
            f"基础 F0: {f0_range} Hz\n"
            f"性格倾向: {temperament}\n"
            f"请分析这只{breed_cn}({pet_type})的情绪状态和需求。"
        ),
        "output": analysis_output,
        "metadata": {
            "pet_type": pet_type,
            "breed": breed_cn,
            "emotion": emotion,
            "context": context,
            "f0_range_hz": f0_range,
            "f0_mean_hz": f0_mean,
            "duration_range_s": duration_range,
            "spectral_centroid_range_hz": centroid_range,
            "voiced_ratio_range": voiced_range,
            "temperament": temperament,
            "weight_kg": weight,
            "lifespan_years": lifespan,
            "genetic_diseases": diseases,
            "ethology_notes": ethology_notes,
            "vocalization_ref": emotion_vocal_map.get("name", "unknown"),
            "vocalization_sources": emotion_vocal_map.get("sources", []),
        },
    }


def _map_emotion_to_vocalization(
    pet_type: str, emotion: str, vocal_list: List[Dict]
) -> Dict:
    """将情绪映射到对应的声学发声类型 (来自行为学研究)"""
    if pet_type == "cat":
        mapping = {
            "hungry": "meow",
            "happy": "purr",
            "angry": "growl",
            "fear": "hiss",
            "seek_attention": "meow",
            "content": "purr",
            "pain": "pain_shriek",
            "alert": "chirp",
            "playful": "trill",
            "sad": "yowl",
        }
    else:
        mapping = {
            "hungry": "bark",
            "happy": "bark",
            "angry": "growl",
            "fear": "whine",
            "seek_attention": "bark",
            "content": "whine",
            "pain": "yip",
            "alert": "bark",
            "playful": "bark",
            "sad": "howl",
        }

    target_name = mapping.get(emotion, "meow" if pet_type == "cat" else "bark")
    for v in vocal_list:
        if v.get("name") == target_name:
            return v
    return {}


def _build_sound_description(
    pet_type: str, breed_cn: str, emotion: str,
    typical_sounds: List[str], vocal_ref: Dict,
) -> str:
    """融合真实品种数据和研究数据构造声音描述"""
    parts = []

    # 品种典型声音 (随机选一个)
    if typical_sounds:
        parts.append(random.choice(typical_sounds))

    # 学术声学参数
    if vocal_ref:
        name_cn = vocal_ref.get("name_cn", "")
        f0 = vocal_ref.get("f0_range_hz")
        duration = vocal_ref.get("typical_duration_s")
        if f0:
            if isinstance(f0, list):
                parts.append(f"学术参考 {name_cn} F0={f0[0]}-{f0[1]} Hz")
            else:
                parts.append(f"学术参考 {name_cn} F0={f0}")
        if duration and isinstance(duration, list):
            parts.append(f"典型时长 {duration[0]}-{duration[1]} s")

    return "；".join(parts) if parts else "普通叫声"


def _build_analysis_output(
    pet_type: str, breed_cn: str, emotion: str, sound_desc: str,
    context: str, f0_range: str, temperament: str, weight: str,
    lifespan: str, diseases: List[str], ethology_notes: str,
    vocal_ref: Dict,
) -> str:
    """构造 LLM 输出 (基于真实数据)"""
    emotion_cn_map = {
        "hungry": "饥饿", "happy": "开心", "angry": "愤怒", "fear": "恐惧",
        "seek_attention": "求关注", "content": "满足", "pain": "疼痛",
        "alert": "警觉", "playful": "想玩耍", "sad": "悲伤",
    }
    emotion_cn = emotion_cn_map.get(emotion, emotion)

    advice_map = {
        "hungry": f"建议检查上次喂食时间。{breed_cn} 体重参考 {weight} kg，按体重计算每日喂食量。",
        "happy": f"{breed_cn} 心情愉悦，可适度互动。性格倾向: {temperament}。",
        "angry": f"立即停止当前行为，给{breed_cn}足够空间，避免直视。等平静后再接近。",
        "fear": f"为{breed_cn}提供安全躲藏空间，保持环境安静，不要强迫互动。",
        "seek_attention": f"回应{breed_cn}需求，互动 5-10 分钟。",
        "content": f"{breed_cn}状态良好，保持当前环境。寿命 {lifespan} 年，注意健康管理。",
        "pain": f"立即检查{breed_cn}身体状况，联系兽医。易患疾病: {', '.join(diseases[:3])}。",
        "alert": f"观察{breed_cn}警觉方向，确认安全后安抚情绪。",
        "playful": f"与{breed_cn}互动玩耍 10-15 分钟。",
        "sad": f"增加{breed_cn}陪伴时间，提供益智玩具。",
    }

    vocal_note = ""
    if vocal_ref:
        name_cn = vocal_ref.get("name_cn", "")
        function = vocal_ref.get("function", "")
        if function:
            vocal_note = f"\n声学功能: {name_cn} 通常用于「{function}」(行为学研究)"

    return (
        f"## 情绪解读\n"
        f"这只{breed_cn}当前情绪状态为「{emotion_cn}」。\n\n"
        f"## 声音分析\n"
        f"声音特征: {sound_desc}\n"
        f"基础 F0 范围: {f0_range} Hz\n"
        f"这种叫声在行为学上对应「{emotion_cn}」状态。{vocal_note}\n\n"
        f"## 情境分析\n"
        f"当前情境: {context}\n"
        f"在{context}情况下，{breed_cn}出现此类声音属于"
        f"{'正常' if emotion in ['happy', 'content', 'seek_attention', 'playful'] else '需要关注'}的表现。\n\n"
        f"## 品种背景\n"
        f"性格倾向: {temperament}\n"
        f"体重参考: {weight} kg\n"
        f"遗传病倾向: {', '.join(diseases[:3]) if diseases else '无明显遗传病'}\n"
        f"动物学笔记: {ethology_notes[:200]}...\n\n"
        f"## 建议措施\n"
        f"{advice_map.get(emotion, '请持续观察宠物状态。')}\n"
    )


def _parse_f0_range(f0_str: str) -> Tuple[float, float]:
    """解析 F0 范围字符串, 返回 (low, high)"""
    try:
        if "-" in str(f0_str):
            parts = str(f0_str).split("-")
            return float(parts[0]), float(parts[1])
        return 400.0, 700.0
    except (ValueError, IndexError):
        return 400.0, 700.0


# =====================================
# 分层抽样划分
# =====================================

def generate_all_samples(
    cat_data: Dict, dog_data: Dict, ethology: Dict,
    samples_per_cell: int = 10,
) -> List[Dict]:
    """
    生成全部样本 (按品种×情绪分层)
    每个 (品种, 情绪) 组合生成 samples_per_cell 条
    """
    all_samples = []

    # 猫: 16 品种 × 10 情绪 × 10 样本 = 1600
    for breed in CAT_BREEDS:
        full_name = _resolve_breed_name(breed)
        breed_info = cat_data.get(full_name)
        if not breed_info:
            print(f"  ⚠ 警告: 猫品种 {breed}(全称:{full_name}) 真实数据缺失, 使用默认")
            breed_info = {
                "breed_cn": breed, "f0_range_hz": "400-700",
                "typical_sounds": ["普通喵叫"], "temperament": "未知",
                "weight_kg": "3-6", "lifespan_years": "12-17",
                "genetic_diseases": [], "ethology_notes": "",
                "acoustic_features": {},
            }

        for emotion in EMOTIONS:
            for ctx_idx in range(samples_per_cell):
                context = CONTEXTS[ctx_idx % len(CONTEXTS)]
                sample = generate_sample(
                    "cat", breed, emotion, context,
                    breed_info, ethology,
                )
                all_samples.append(sample)

    # 狗: 16 品种 × 10 情绪 × 10 样本 = 1600
    for breed in DOG_BREEDS:
        full_name = _resolve_breed_name(breed)
        breed_info = dog_data.get(full_name)
        if not breed_info:
            print(f"  ⚠ 警告: 狗品种 {breed}(全称:{full_name}) 真实数据缺失, 使用默认")
            breed_info = {
                "breed_cn": breed, "f0_range_hz": "200-500",
                "typical_sounds": ["普通吠叫"], "temperament": "未知",
                "weight_kg": "10-30", "lifespan_years": "10-14",
                "genetic_diseases": [], "ethology_notes": "",
                "acoustic_features": {},
            }

        for emotion in EMOTIONS:
            for ctx_idx in range(samples_per_cell):
                context = CONTEXTS[ctx_idx % len(CONTEXTS)]
                sample = generate_sample(
                    "dog", breed, emotion, context,
                    breed_info, ethology,
                )
                all_samples.append(sample)

    return all_samples


def stratified_split(
    samples: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    分层抽样划分
    按 (pet_type, breed, emotion) 三元组分层
    每层内部按比例划分 train/val/test
    """
    # 按层分组
    layers: Dict[str, List[Dict]] = defaultdict(list)
    for s in samples:
        key = f"{s['metadata']['pet_type']}|{s['metadata']['breed']}|{s['metadata']['emotion']}"
        layers[key].append(s)

    train, val, test = [], [], []

    for layer_key, layer_samples in layers.items():
        # 随机打乱
        random.shuffle(layer_samples)
        n = len(layer_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # n_test = n - n_train - n_val (剩余给 test)

        train.extend(layer_samples[:n_train])
        val.extend(layer_samples[n_train:n_train + n_val])
        test.extend(layer_samples[n_train + n_val:])

    # 全局打乱
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# =====================================
# 主流程
# =====================================

def build_dataset():
    """主流程: 加载真实数据 → 生成样本 → 分层划分 → 保存"""
    os.makedirs(DATASET_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  数据集划分: 基于真实动物学数据生成 train/val/test")
    print("=" * 60)

    # 1. 加载真实数据
    print("\n[1/5] 加载爬取的真实动物学数据...")
    cat_data, dog_data = load_real_breed_data()
    ethology = load_ethology_data()
    print(f"  ✓ 猫品种: {len(cat_data)} 个")
    print(f"  ✓ 狗品种: {len(dog_data)} 个")
    print(f"  ✓ 声学研究: {len(ethology)} 个主题")

    # 2. 生成全部样本
    print("\n[2/5] 基于真实数据生成训练样本...")
    samples = generate_all_samples(cat_data, dog_data, ethology, samples_per_cell=10)
    print(f"  ✓ 总样本数: {len(samples)}")
    # 统计
    cat_n = sum(1 for s in samples if s["metadata"]["pet_type"] == "cat")
    dog_n = sum(1 for s in samples if s["metadata"]["pet_type"] == "dog")
    print(f"  ✓ 猫样本: {cat_n}  |  狗样本: {dog_n}")

    # 3. 分层划分
    print("\n[3/5] 分层抽样划分 (80/10/10)...")
    train, val, test = stratified_split(samples)
    print(f"  ✓ train: {len(train)} 条 ({len(train)/len(samples)*100:.1f}%)")
    print(f"  ✓ val:   {len(val)} 条 ({len(val)/len(samples)*100:.1f}%)")
    print(f"  ✓ test:  {len(test)} 条 ({len(test)/len(samples)*100:.1f}%)")

    # 4. 保存
    print("\n[4/5] 保存数据集...")
    train_path = os.path.join(DATASET_DIR, "train.json")
    val_path = os.path.join(DATASET_DIR, "val.json")
    test_path = os.path.join(DATASET_DIR, "test.json")

    for path, data in [(train_path, train), (val_path, val), (test_path, test)]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {path} ({len(data)} 条)")

    # 5. 生成统计报告
    print("\n[5/5] 生成统计报告...")
    stats = {
        "total_samples": len(samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "ratio": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "cat_breeds": len(CAT_BREEDS),
        "dog_breeds": len(DOG_BREEDS),
        "emotions": len(EMOTIONS),
        "random_seed": RANDOM_SEED,
        "data_sources": {
            "cat_breeds": "data/crawler/cat_breeds_real.json",
            "dog_breeds": "data/crawler/dog_breeds_real.json",
            "ethology": "data/crawler/ethology_research.json",
        },
        "split_strategy": "stratified by (pet_type, breed, emotion)",
    }
    stats_path = os.path.join(DATASET_DIR, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  ✓ {stats_path}")

    print("\n" + "=" * 60)
    print("  数据集划分完成!")
    print("=" * 60)
    print(f"\n使用方法:")
    print(f"  from datasets import load_dataset")
    print(f"  train = json.load(open('{train_path}'))")
    print(f"  val = json.load(open('{val_path}'))")
    print(f"  test = json.load(open('{test_path}'))")

    return train, val, test, stats


if __name__ == "__main__":
    build_dataset()
