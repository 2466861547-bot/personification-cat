"""
宠物知识库: 构建和管理宠物行为/声音知识库
包含品种特征、声音含义、情绪解读、行为指南等
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeEntry:
    """知识条目"""
    pet_type: str       # cat / dog
    breed: str          # 品种
    emotion: str       # 情绪
    sound_description: str   # 声音描述
    meaning: str        # 含义解读
    context: str        # 情境
    advice: str         # 建议措施
    source: str = "expert"   # 来源


# 预置知识库数据
DEFAULT_KNOWLEDGE = [
    # ===== 猫咪 =====
    KnowledgeEntry("cat", "通用", "hungry",
        "短促而高频的喵叫，通常重复2-3次，音调上扬",
        "猫咪感到饥饿，请求食物",
        "通常在饭点前后，或看到食物相关物品时",
        "检查上次喂食时间，若超过4小时可适量喂食；避免过度喂食"),
    KnowledgeEntry("cat", "通用", "happy",
        "轻柔的短喵叫，伴随呼噜声，音调平稳",
        "猫咪心情愉悦，感到安全和满足",
        "被抚摸、晒太阳、或在舒适环境中",
        "继续当前互动，可适度抚摸猫咪头部和下巴"),
    KnowledgeEntry("cat", "通用", "angry",
        "低沉的嘶嘶声(hissing)或低吼(growling)，音调下降",
        "猫咪感到威胁或愤怒，警告对方远离",
        "遇到陌生人/动物、被强迫做不喜欢的事、领地被侵入",
        "立即停止当前行为，给猫咪空间，不要直视猫眼"),
    KnowledgeEntry("cat", "通用", "fear",
        "尖锐的哀嚎或长鸣，声音颤抖，可能伴随炸毛",
        "猫咪感到恐惧或极度不安",
        "雷雨天气、陌生环境、巨响、被带去宠物医院",
        "提供躲藏空间(如猫窝/纸箱)，保持安静，不要强迫互动"),
    KnowledgeEntry("cat", "通用", "seek_attention",
        "持续且有节奏的喵叫，可能伴随蹭腿、扒门",
        "猫咪想要引起注意，寻求互动或服务",
        "主人回家、长时间未互动、想进出房间",
        "回应猫咪，进行短暂互动(抚摸/玩耍5-10分钟)"),
    KnowledgeEntry("cat", "通用", "content",
        "持续低频的呼噜声(purring)，约25-150Hz",
        "猫咪感到满足和放松，也可能在自我疗愈",
        "舒适环境中、被抚摸时、入睡前后",
        "保持环境安静，享受与猫咪的亲密时光"),
    KnowledgeEntry("cat", "通用", "pain",
        "异常的尖叫或持续哀鸣，声音尖锐且不规律",
        "猫咪可能受伤或感到疼痛，需要关注",
        "摔倒后、手术后、泌尿问题、关节疼痛",
        "立即检查猫咪身体状况，联系兽医进行诊断"),
    KnowledgeEntry("cat", "通用", "alert",
        "短促高频的咯咯声(chattering)或尖锐喵叫",
        "猫咪发现猎物或感兴趣的目标，处于兴奋状态",
        "看到窗外鸟类/昆虫，或发现移动的小物体",
        "观察猫咪视线方向，可提供逗猫棒等玩具替代"),

    # ===== 狗狗 =====
    KnowledgeEntry("dog", "通用", "hungry",
        "短促的汪汪叫2-3声，可能伴随舔食盆、在食物区徘徊",
        "狗狗感到饥饿，请求食物",
        "饭点前后，或看到/闻到食物",
        "检查喂食计划，按量喂食；避免喂人类食物"),
    KnowledgeEntry("dog", "通用", "happy",
        "轻快的汪汪叫1-2声，伴随摇尾巴、身体放松",
        "狗狗心情愉快，想要互动或玩耍",
        "主人回家、看到玩具、准备出门散步",
        "回应狗狗，进行互动游戏(扔球/拔河)"),
    KnowledgeEntry("dog", "通用", "angry",
        "低沉的咆哮(growling)，音调持续下降，可能露牙",
        "狗狗感到威胁或不悦，警告对方停止",
        "护食、被陌生人靠近、领地被侵入",
        "立即停止接近，不要对视，缓慢后退给狗狗空间"),
    KnowledgeEntry("dog", "通用", "fear",
        "尖锐的哀嚎或呜咽(whining)，声音持续且颤抖",
        "狗狗感到恐惧或焦虑",
        "雷雨天气、独处、陌生环境、听到巨响",
        "安抚狗狗，提供安全空间(狗窝)，考虑脱敏训练"),
    KnowledgeEntry("dog", "通用", "alert",
        "连续快速的汪汪叫，音调较高，间隔短",
        "狗狗警觉到异常情况，发出警报",
        "有人按门铃、听到异常声响、发现陌生人/动物",
        "查看情况，确认安全后安抚狗狗停止吠叫"),
    KnowledgeEntry("dog", "通用", "playful",
        "活泼的汪汪叫，伴随前肢下压(play bow)、摇尾巴",
        "狗狗想玩耍，邀请互动",
        "看到主人/其他狗、有玩具时",
        "与狗狗互动玩耍10-15分钟，满足其运动需求"),
    KnowledgeEntry("dog", "通用", "sad",
        "低沉的呜咽声，持续而缓慢，伴随耳朵下垂",
        "狗狗感到失落或孤独",
        "主人离开、被忽视、同伴分离",
        "增加陪伴时间，可考虑提供益智玩具"),
    KnowledgeEntry("dog", "通用", "pain",
        "突然的尖叫或不寻常的低吟，不愿被触碰",
        "狗狗可能受伤或感到疼痛",
        "运动后、跳跃后、老年犬关节问题",
        "检查受伤部位，避免触碰痛处，联系兽医"),

    # ===== 品种特定声音特征 (16 猫 + 16 狗) =====
    # === 猫 ===
    KnowledgeEntry("cat", "英短", "hungry",
        "英短特有的低沉短喵，声音比其他品种更浑厚",
        "英短饿了，但因性格沉稳，叫声较含蓄",
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


class PetKnowledgeBase:
    """宠物知识库管理器"""

    def __init__(self, knowledge_dir: str = "./data/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.entries: List[KnowledgeEntry] = []
        os.makedirs(knowledge_dir, exist_ok=True)
        self._load_default_knowledge()

    def _load_default_knowledge(self):
        """加载默认知识"""
        self.entries = DEFAULT_KNOWLEDGE.copy()

    def add_entry(self, entry: KnowledgeEntry):
        """添加知识条目"""
        self.entries.append(entry)

    def add_entries(self, entries: List[KnowledgeEntry]):
        """批量添加"""
        self.entries.extend(entries)

    def save_to_json(self, file_path: str = None):
        """保存知识库到 JSON 文件"""
        if file_path is None:
            file_path = os.path.join(self.knowledge_dir, "pet_knowledge.json")

        data = []
        for entry in self.entries:
            data.append({
                "pet_type": entry.pet_type,
                "breed": entry.breed,
                "emotion": entry.emotion,
                "sound_description": entry.sound_description,
                "meaning": entry.meaning,
                "context": entry.context,
                "advice": entry.advice,
                "source": entry.source,
            })

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"知识库已保存: {file_path} ({len(data)} 条)")

    def load_from_json(self, file_path: str):
        """从 JSON 加载知识库"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.entries = []
        for item in data:
            self.entries.append(KnowledgeEntry(**item))
        print(f"已加载知识库: {file_path} ({len(self.entries)} 条)")

    def search(
        self,
        pet_type: str = None,
        emotion: str = None,
        breed: str = None,
    ) -> List[KnowledgeEntry]:
        """条件搜索"""
        results = self.entries
        if pet_type:
            results = [e for e in results if e.pet_type == pet_type]
        if emotion:
            results = [e for e in results if e.emotion == emotion]
        if breed:
            results = [e for e in results if e.breed == breed or e.breed == "通用"]
        return results

    def to_documents(self) -> List[Dict[str, str]]:
        """转换为文档格式(供向量检索使用)"""
        documents = []
        for entry in self.entries:
            text = (
                f"宠物类型: {entry.pet_type}\n"
                f"品种: {entry.breed}\n"
                f"情绪: {entry.emotion}\n"
                f"声音描述: {entry.sound_description}\n"
                f"含义: {entry.meaning}\n"
                f"情境: {entry.context}\n"
                f"建议: {entry.advice}\n"
            )
            documents.append({
                "text": text,
                "metadata": {
                    "pet_type": entry.pet_type,
                    "breed": entry.breed,
                    "emotion": entry.emotion,
                },
            })
        return documents

    def export_for_training(self) -> List[Dict]:
        """导出为 LLM 训练数据格式"""
        training_data = []
        for entry in self.entries:
            # 指令微调格式
            training_data.append({
                "system": "你是一个宠物语言翻译专家，能够根据宠物声音特征判断其情绪和需求。",
                "input": (
                    f"宠物类型: {entry.pet_type}，品种: {entry.breed}\n"
                    f"声音特征: {entry.sound_description}\n"
                    f"情境: {entry.context}\n"
                    f"请分析这只{entry.pet_type}的情绪和需求，并给出建议。"
                ),
                "output": (
                    f"## 情绪解读\n"
                    f"这只{entry.pet_type}当前的情绪是「{entry.emotion}」。\n\n"
                    f"## 声音分析\n"
                    f"{entry.sound_description}\n\n"
                    f"## 含义\n"
                    f"{entry.meaning}\n\n"
                    f"## 情境分析\n"
                    f"{entry.context}\n\n"
                    f"## 建议措施\n"
                    f"{entry.advice}\n"
                ),
            })
        return training_data
