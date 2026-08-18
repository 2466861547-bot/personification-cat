"""
增强知识库模块: 基于爬取的真实动物学数据扩充知识库条目
- 输入: data/crawler/*.json 真实数据
- 输出: data/knowledge/expanded_knowledge.json 扩充知识库
- 集成: 通过 PetKnowledgeBase.add_entries() 合并到现有知识库
"""

import os
import json
from typing import List, Dict
from dataclasses import asdict

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CRAWLER_DIR = os.path.join(DATA_DIR, "crawler")
KNOWLEDGE_DIR = os.path.join(DATA_DIR, "knowledge")

# 引入项目知识库类
import sys
sys.path.insert(0, PROJECT_ROOT)
from src.rag.knowledge_base import KnowledgeEntry


# =====================================
# 基于真实品种数据生成知识条目
# =====================================

def build_breed_knowledge_from_real_data() -> List[Dict]:
    """基于爬取的真实品种数据生成知识条目 (每品种×情绪)"""
    entries = []

    # 加载真实品种数据
    cat_path = os.path.join(CRAWLER_DIR, "cat_breeds_real.json")
    dog_path = os.path.join(CRAWLER_DIR, "dog_breeds_real.json")

    cat_breeds = _load_json(cat_path)
    dog_breeds = _load_json(dog_path)

    # 10 种情绪及其对应声学和行为描述
    emotion_meta = {
        "hungry": {"cat_vocal": "meow", "dog_vocal": "bark",
                   "meaning_cn": "饥饿，请求食物"},
        "happy": {"cat_vocal": "purr", "dog_vocal": "bark",
                  "meaning_cn": "心情愉悦，感到满足"},
        "angry": {"cat_vocal": "growl", "dog_vocal": "growl",
                  "meaning_cn": "愤怒，警告对方远离"},
        "fear": {"cat_vocal": "hiss", "dog_vocal": "whine",
                 "meaning_cn": "恐惧，感到不安"},
        "seek_attention": {"cat_vocal": "meow", "dog_vocal": "bark",
                          "meaning_cn": "求关注，寻求互动"},
        "content": {"cat_vocal": "purr", "dog_vocal": "whine",
                    "meaning_cn": "满足和放松"},
        "pain": {"cat_vocal": "pain_shriek", "dog_vocal": "yip",
                 "meaning_cn": "受伤或疼痛"},
        "alert": {"cat_vocal": "chirp", "dog_vocal": "bark",
                 "meaning_cn": "警觉，发现目标"},
        "playful": {"cat_vocal": "trill", "dog_vocal": "bark",
                   "meaning_cn": "想玩耍，邀请互动"},
        "sad": {"cat_vocal": "yowl", "dog_vocal": "howl",
               "meaning_cn": "失落或孤独"},
    }

    # 猫品种知识条目
    for breed in cat_breeds:
        breed_cn = breed.get("breed_cn", "未知")
        f0_range = breed.get("f0_range_hz", "400-700")
        typical_sounds = breed.get("typical_sounds", ["普通喵叫"])
        temperament = breed.get("temperament", "未知")
        diseases = breed.get("genetic_diseases", [])
        ethology_notes = breed.get("ethology_notes", "")
        sources = breed.get("sources", [])

        for emotion, meta in emotion_meta.items():
            sound_desc = _build_sound_desc(
                breed_cn, f0_range, typical_sounds, meta["cat_vocal"]
            )
            meaning = f"{breed_cn} 当前{meta['meaning_cn']}。性格: {temperament[:50]}"
            context = _emotion_context(emotion)
            advice = _build_advice(breed_cn, emotion, diseases)

            entry = KnowledgeEntry(
                pet_type="cat",
                breed=breed_cn,
                emotion=emotion,
                sound_description=sound_desc,
                meaning=meaning,
                context=context,
                advice=advice,
                source="real_crawler_data",
            )
            entry_dict = asdict(entry)
            entry_dict["f0_range_hz"] = f0_range
            entry_dict["temperament"] = temperament
            entry_dict["genetic_diseases"] = diseases
            entry_dict["ethology_notes"] = ethology_notes[:300]
            entry_dict["sources"] = sources
            entries.append(entry_dict)

    # 狗品种知识条目
    for breed in dog_breeds:
        breed_cn = breed.get("breed_cn", "未知")
        f0_range = breed.get("f0_range_hz", "200-500")
        typical_sounds = breed.get("typical_sounds", ["普通吠叫"])
        temperament = breed.get("temperament", "未知")
        diseases = breed.get("genetic_diseases", [])
        ethology_notes = breed.get("ethology_notes", "")
        sources = breed.get("sources", [])

        for emotion, meta in emotion_meta.items():
            sound_desc = _build_sound_desc(
                breed_cn, f0_range, typical_sounds, meta["dog_vocal"]
            )
            meaning = f"{breed_cn} 当前{meta['meaning_cn']}。性格: {temperament[:50]}"
            context = _emotion_context(emotion)
            advice = _build_advice(breed_cn, emotion, diseases)

            entry = KnowledgeEntry(
                pet_type="dog",
                breed=breed_cn,
                emotion=emotion,
                sound_description=sound_desc,
                meaning=meaning,
                context=context,
                advice=advice,
                source="real_crawler_data",
            )
            entry_dict = asdict(entry)
            entry_dict["f0_range_hz"] = f0_range
            entry_dict["temperament"] = temperament
            entry_dict["genetic_diseases"] = diseases
            entry_dict["ethology_notes"] = ethology_notes[:300]
            entry_dict["sources"] = sources
            entries.append(entry_dict)

    return entries


def _build_sound_desc(breed: str, f0_range: str, typical_sounds: List[str], vocal_type: str) -> str:
    """构造声音描述"""
    sound = typical_sounds[0] if typical_sounds else "普通叫声"
    return f"{breed}的{vocal_type}，F0={f0_range}Hz；典型: {sound}"


def _emotion_context(emotion: str) -> str:
    """情绪对应的典型情境"""
    return {
        "hungry": "饭点前后，或看到/闻到食物时",
        "happy": "主人回家、被抚摸、晒太阳、玩耍时",
        "angry": "遇到陌生人/动物、被强迫、领地被侵入时",
        "fear": "雷雨天气、陌生环境、听到巨响时",
        "seek_attention": "主人回家、长时间未互动、想进出房间时",
        "content": "舒适环境中、被抚摸时、入睡前后",
        "pain": "摔倒后、手术后、关节疼痛时",
        "alert": "看到窗外鸟类/昆虫、听到异常声响时",
        "playful": "看到主人/玩具、想互动时",
        "sad": "主人离开、被忽视、同伴分离时",
    }.get(emotion, "多种情境")


def _build_advice(breed: str, emotion: str, diseases: List[str]) -> str:
    """构造建议措施"""
    base_advice = {
        "hungry": f"检查{breed}上次喂食时间，按体重适量喂食",
        "happy": f"{breed}心情愉悦，可适度互动",
        "angry": f"立即停止当前行为，给{breed}空间，避免直视",
        "fear": f"为{breed}提供安全空间，保持安静",
        "seek_attention": f"回应{breed}，互动 5-10 分钟",
        "content": f"{breed}状态良好，保持当前环境",
        "pain": f"立即检查{breed}身体，联系兽医",
        "alert": f"观察{breed}警觉方向，确认安全后安抚",
        "playful": f"与{breed}互动玩耍 10-15 分钟",
        "sad": f"增加{breed}陪伴时间，提供益智玩具",
    }.get(emotion, "请持续观察宠物状态")

    if diseases and emotion in ["pain", "fear", "alert"]:
        return f"{base_advice}。易患疾病: {', '.join(diseases[:3])}，注意排查"

    return base_advice


def build_ethology_research_knowledge() -> List[Dict]:
    """基于声学行为学研究数据生成知识条目"""
    entries = []

    path = os.path.join(CRAWLER_DIR, "ethology_research.json")
    if not os.path.exists(path):
        return entries

    ethology = _load_json(path)

    # 猫咪声学知识
    cat_acoustics = ethology.get("cat_acoustics", {})
    for vocal in cat_acoustics.get("vocalizations", []):
        name = vocal.get("name", "")
        name_cn = vocal.get("name_cn", "")
        f0 = vocal.get("f0_range_hz", [])
        duration = vocal.get("typical_duration_s", [])
        function = vocal.get("function", "")
        contexts = vocal.get("contexts", [])
        sources = vocal.get("sources", [])

        f0_str = f"{f0[0]}-{f0[1]} Hz" if isinstance(f0, list) and len(f0) == 2 else str(f0)
        dur_str = f"{duration[0]}-{duration[1]} s" if isinstance(duration, list) and len(duration) == 2 else str(duration)

        sound_desc = f"{name_cn}({name})，F0={f0_str}，时长={dur_str}"
        meaning = f"行为学功能: {function}" if function else "行为学发声类型"
        context = "、".join(contexts[:3]) if contexts else "多种情境"

        entry = KnowledgeEntry(
            pet_type="cat",
            breed="学术研究",
            emotion="general",
            sound_description=sound_desc,
            meaning=meaning,
            context=context,
            advice="参考学术论文: " + ", ".join(sources[:2]) if sources else "无",
            source="ethology_research",
        )
        entry_dict = asdict(entry)
        entry_dict["f0_range_hz"] = f0_str
        entry_dict["vocalization_name"] = name
        entry_dict["sources"] = sources
        entries.append(entry_dict)

    # 狗狗声学知识
    dog_acoustics = ethology.get("dog_acoustics", {})
    for vocal in dog_acoustics.get("vocalizations", []):
        name = vocal.get("name", "")
        name_cn = vocal.get("name_cn", "")
        f0 = vocal.get("f0_range_hz", [])
        duration = vocal.get("typical_duration_s", [])
        function = vocal.get("function", "")
        contexts = vocal.get("contexts", [])
        sources = vocal.get("sources", [])

        f0_str = f"{f0[0]}-{f0[1]} Hz" if isinstance(f0, list) and len(f0) == 2 else str(f0)
        dur_str = f"{duration[0]}-{duration[1]} s" if isinstance(duration, list) and len(duration) == 2 else str(duration)

        sound_desc = f"{name_cn}({name})，F0={f0_str}，时长={dur_str}"
        meaning = f"行为学功能: {function}" if function else "行为学发声类型"
        context = "、".join(contexts[:3]) if contexts else "多种情境"

        entry = KnowledgeEntry(
            pet_type="dog",
            breed="学术研究",
            emotion="general",
            sound_description=sound_desc,
            meaning=meaning,
            context=context,
            advice="参考学术论文: " + ", ".join(sources[:2]) if sources else "无",
            source="ethology_research",
        )
        entry_dict = asdict(entry)
        entry_dict["f0_range_hz"] = f0_str
        entry_dict["vocalization_name"] = name
        entry_dict["sources"] = sources
        entries.append(entry_dict)

    # 情绪映射知识
    emotion_mapping = ethology.get("emotion_mapping", {})
    if isinstance(emotion_mapping, dict):
        # cat_emotion_vocalization_map 和 dog_emotion_vocalization_map 是列表
        cat_map = emotion_mapping.get("cat_emotion_vocalization_map", [])
        dog_map = emotion_mapping.get("dog_emotion_vocalization_map", [])
        theoretical = emotion_mapping.get("theoretical_framework", {})

        # 理论框架条目
        for framework_name, framework_desc in theoretical.items():
            entry = KnowledgeEntry(
                pet_type="general",
                breed="情绪理论",
                emotion="theoretical",
                sound_description=f"理论框架: {framework_name}",
                meaning=str(framework_desc)[:200] if isinstance(framework_desc, (dict, list)) else str(framework_desc)[:200],
                context="跨物种通用",
                advice=f"参考理论: {framework_name}",
                source="ethology_research",
            )
            entries.append(asdict(entry))

        # 猫情绪映射条目
        for em in cat_map:
            if not isinstance(em, dict):
                continue
            emotion = em.get("emotion", "")
            vocalizations = em.get("vocalizations", [])
            acoustic_cues = em.get("acoustic_cues", [])
            valence = em.get("valence", "")
            arousal = em.get("arousal", "")

            entry = KnowledgeEntry(
                pet_type="cat",
                breed="情绪理论",
                emotion=emotion,
                sound_description=", ".join(vocalizations) if vocalizations else "未知",
                meaning=f"效价: {valence}, 唤醒度: {arousal}",
                context="、".join(acoustic_cues) if acoustic_cues else "多种情境",
                advice="参考猫情绪-发声映射研究",
                source="ethology_research",
            )
            entries.append(asdict(entry))

        # 狗情绪映射条目
        for em in dog_map:
            if not isinstance(em, dict):
                continue
            emotion = em.get("emotion", "")
            vocalizations = em.get("vocalizations", [])
            acoustic_cues = em.get("acoustic_cues", [])
            valence = em.get("valence", "")
            arousal = em.get("arousal", "")

            entry = KnowledgeEntry(
                pet_type="dog",
                breed="情绪理论",
                emotion=emotion,
                sound_description=", ".join(vocalizations) if vocalizations else "未知",
                meaning=f"效价: {valence}, 唤醒度: {arousal}",
                context="、".join(acoustic_cues) if acoustic_cues else "多种情境",
                advice="参考狗情绪-发声映射研究",
                source="ethology_research",
            )
            entries.append(asdict(entry))

    return entries


def build_public_dataset_knowledge() -> List[Dict]:
    """基于公开数据集信息生成知识条目"""
    entries = []

    path = os.path.join(CRAWLER_DIR, "ethology_research.json")
    if not os.path.exists(path):
        return entries

    ethology = _load_json(path)
    for ds in ethology.get("public_datasets", []):
        entry = KnowledgeEntry(
            pet_type="general",
            breed="公开数据集",
            emotion="metadata",
            sound_description=f"{ds.get('name', '')} - {ds.get('samples', 0)} 样本",
            meaning=f"可用作训练补充数据，宠物相关样本 {ds.get('pet_sounds', 0)} 条",
            context=ds.get("description", ""),
            advice=f"数据集 URL: {ds.get('url', 'unknown')}",
            source="public_dataset",
        )
        entries.append(asdict(entry))

    return entries


def _load_json(path: str) -> Dict:
    """加载 JSON 文件"""
    if not os.path.exists(path):
        return {} if path.endswith("ethology_research.json") else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =====================================
# 主流程
# =====================================

def build_expanded_knowledge():
    """主流程: 生成扩充知识库"""
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    print("=" * 60)
    print("  扩充知识库: 基于真实动物学数据")
    print("=" * 60)

    # 1. 品种特定知识
    print("\n[1/3] 生成品种特定知识条目...")
    breed_entries = build_breed_knowledge_from_real_data()
    print(f"  ✓ 品种知识条目: {len(breed_entries)} 条")

    # 2. 声学行为学知识
    print("\n[2/3] 生成声学行为学研究知识条目...")
    ethology_entries = build_ethology_research_knowledge()
    print(f"  ✓ 行为学知识条目: {len(ethology_entries)} 条")

    # 3. 公开数据集知识
    print("\n[3/3] 生成公开数据集知识条目...")
    dataset_entries = build_public_dataset_knowledge()
    print(f"  ✓ 数据集知识条目: {len(dataset_entries)} 条")

    # 合并保存
    all_entries = breed_entries + ethology_entries + dataset_entries
    output_path = os.path.join(KNOWLEDGE_DIR, "expanded_knowledge.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"  扩充知识库完成!")
    print(f"  总条目: {len(all_entries)} 条")
    print(f"  保存到: {output_path}")
    print("=" * 60)

    # 统计
    stats = {
        "total": len(all_entries),
        "breed_specific": len(breed_entries),
        "ethology_research": len(ethology_entries),
        "public_datasets": len(dataset_entries),
        "cat_breeds_covered": len(set(e["breed"] for e in breed_entries if e["pet_type"] == "cat")),
        "dog_breeds_covered": len(set(e["breed"] for e in breed_entries if e["pet_type"] == "dog")),
    }
    stats_path = os.path.join(KNOWLEDGE_DIR, "knowledge_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  统计: {stats}")

    return all_entries


if __name__ == "__main__":
    build_expanded_knowledge()
