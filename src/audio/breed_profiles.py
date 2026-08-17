"""
品种声学特征库 + 品种识别器
================================================
通过 F0 基频、单次发声时长、频谱质心、浊音比例等特征区分不同品种的猫狗声音。

设计原理
--------
1. 体型差异 → 共鸣腔大小不同 → 基频(F0)差异
   - 大型猫狗(缅因/阿拉斯加): F0 低 (200-420 Hz)
   - 小型猫狗(矮脚拿破仑/博美): F0 高 (450-850 Hz)
2. 喉部结构 → 声带长度与张力 → 音色(频谱质心)差异
   - 长头型(暹罗/灵缇): 高频明亮 (1500-2800 Hz)
   - 短头型(加菲/法斗): 低频沉闷 (600-1500 Hz)
3. 行为习惯 → 发声时长
   - 话痨型(暹罗/哈士奇): 单次发声 0.8-3.5s
   - 沉默型(英短/柴犬): 0.15-0.45s
4. 品种特化声音 → signature 标签
   - 缅因 chirp_trill, 哈士奇 howl_yodel, 暹罗 loud_long_meow
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BreedAcousticProfile:
    """品种声学特征档案"""
    breed: str
    pet_type: str
    f0_range: Tuple[float, float]                   # 基频范围 (Hz)
    typical_duration: Tuple[float, float]          # 典型单次发声时长 (秒)
    spectral_centroid_range: Tuple[float, float]    # 频谱质心范围 (Hz)
    voiced_ratio_range: Tuple[float, float]        # 浊音比例
    purr_freq: Optional[Tuple[float, float]] = None  # 猫呼噜频率 (Hz)
    bark_rate: Optional[Tuple[float, float]] = None  # 狗吠叫频率 (次/分钟)
    signature: str = ""        # 声音特征签名
    description: str = ""      # 声音特征描述


# =====================================
# 品种声学特征库 (16 猫 + 16 狗)
# =====================================

BREED_PROFILES: Dict[str, BreedAcousticProfile] = {
    # ============ 猫 (16 种) ============
    "英短": BreedAcousticProfile(
        breed="英短", pet_type="cat",
        f0_range=(300, 500),
        typical_duration=(0.15, 0.35),
        spectral_centroid_range=(900, 1800),
        voiced_ratio_range=(0.55, 0.80),
        purr_freq=(22, 28),
        signature="low_short_meow",
        description="性格沉稳少叫，叫声低沉短促浑厚",
    ),
    "美短": BreedAcousticProfile(
        breed="美短", pet_type="cat",
        f0_range=(350, 580),
        typical_duration=(0.20, 0.45),
        spectral_centroid_range=(1000, 2000),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(23, 28),
        signature="medium_meow",
        description="活泼好动，叫声适中清脆",
    ),
    "布偶": BreedAcousticProfile(
        breed="布偶", pet_type="cat",
        f0_range=(320, 520),
        typical_duration=(0.20, 0.50),
        spectral_centroid_range=(950, 1900),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(20, 26),
        signature="soft_sweet_meow",
        description="性格温顺，叫声轻柔甜美",
    ),
    "橘猫": BreedAcousticProfile(
        breed="橘猫", pet_type="cat",
        f0_range=(380, 620),
        typical_duration=(0.25, 0.60),
        spectral_centroid_range=(1100, 2100),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(22, 30),
        signature="loud_food_meow",
        description="贪吃活泼，吃饭时叫声大而频繁",
    ),
    "暹罗": BreedAcousticProfile(
        breed="暹罗", pet_type="cat",
        f0_range=(520, 850),
        typical_duration=(0.40, 1.20),
        spectral_centroid_range=(1500, 2800),
        voiced_ratio_range=(0.70, 0.95),
        purr_freq=(24, 30),
        signature="loud_long_meow",
        description="话痨猫，叫声大而长，频率高，被誉为猫中歌唱家",
    ),
    "波斯": BreedAcousticProfile(
        breed="波斯", pet_type="cat",
        f0_range=(280, 480),
        typical_duration=(0.15, 0.40),
        spectral_centroid_range=(800, 1700),
        voiced_ratio_range=(0.55, 0.80),
        purr_freq=(20, 26),
        signature="soft_low_meow",
        description="安静优雅，叫声轻柔低沉",
    ),
    "缅因": BreedAcousticProfile(
        breed="缅因", pet_type="cat",
        f0_range=(220, 420),
        typical_duration=(0.15, 0.45),
        spectral_centroid_range=(900, 1800),
        voiced_ratio_range=(0.60, 0.90),
        purr_freq=(20, 26),
        signature="chirp_trill",
        description="大型猫，以啁啾颤音(chirp/trill)著称，F0 较低",
    ),
    "狸花": BreedAcousticProfile(
        breed="狸花", pet_type="cat",
        f0_range=(380, 620),
        typical_duration=(0.20, 0.50),
        spectral_centroid_range=(1100, 2100),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(24, 30),
        signature="sharp_meow",
        description="本土敏捷猫，叫声清亮有穿透力",
    ),
    "矮脚拿破仑": BreedAcousticProfile(
        breed="矮脚拿破仑", pet_type="cat",
        f0_range=(450, 750),
        typical_duration=(0.20, 0.45),
        spectral_centroid_range=(1200, 2400),
        voiced_ratio_range=(0.55, 0.85),
        purr_freq=(22, 30),
        signature="soft_high_trill",
        description="小型猫(Munchkin×Persian)，腿短共鸣腔小→F0偏高；"
                    "波斯血统→叫声轻柔；带颤音(trill)，性格温顺亲人",
    ),
    "苏格兰折耳": BreedAcousticProfile(
        breed="苏格兰折耳", pet_type="cat",
        f0_range=(350, 580),
        typical_duration=(0.20, 0.50),
        spectral_centroid_range=(1000, 2000),
        voiced_ratio_range=(0.55, 0.85),
        purr_freq=(22, 28),
        signature="soft_medium_meow",
        description="安静温和，叫声中等柔和",
    ),
    "阿比西尼亚": BreedAcousticProfile(
        breed="阿比西尼亚", pet_type="cat",
        f0_range=(420, 700),
        typical_duration=(0.20, 0.55),
        spectral_centroid_range=(1200, 2300),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(24, 30),
        signature="clear_active_meow",
        description="活跃好动，叫声清脆明亮",
    ),
    "孟加拉豹猫": BreedAcousticProfile(
        breed="孟加拉豹猫", pet_type="cat",
        f0_range=(400, 720),
        typical_duration=(0.25, 0.70),
        spectral_centroid_range=(1200, 2400),
        voiced_ratio_range=(0.60, 0.88),
        purr_freq=(24, 30),
        signature="wild_chirp",
        description="野性血统，叫声多变，常发啁啾音",
    ),
    "斯芬克斯无毛猫": BreedAcousticProfile(
        breed="斯芬克斯无毛猫", pet_type="cat",
        f0_range=(420, 700),
        typical_duration=(0.25, 0.65),
        spectral_centroid_range=(1200, 2300),
        voiced_ratio_range=(0.60, 0.88),
        purr_freq=(24, 32),
        signature="loud_meow",
        description="性格外向，叫声大而频繁",
    ),
    "俄罗斯蓝猫": BreedAcousticProfile(
        breed="俄罗斯蓝猫", pet_type="cat",
        f0_range=(320, 520),
        typical_duration=(0.15, 0.40),
        spectral_centroid_range=(900, 1800),
        voiced_ratio_range=(0.55, 0.80),
        purr_freq=(20, 26),
        signature="quiet_low_meow",
        description="安静害羞，叫声低而少",
    ),
    "加菲猫": BreedAcousticProfile(
        breed="加菲猫", pet_type="cat",
        f0_range=(280, 480),
        typical_duration=(0.15, 0.40),
        spectral_centroid_range=(800, 1600),
        voiced_ratio_range=(0.50, 0.78),
        purr_freq=(20, 26),
        signature="nasal_low_meow",
        description="异国短毛猫，短鼻鼻腔共鸣→鼻音重、F0 偏低",
    ),
    "美国卷耳猫": BreedAcousticProfile(
        breed="美国卷耳猫", pet_type="cat",
        f0_range=(360, 600),
        typical_duration=(0.20, 0.50),
        spectral_centroid_range=(1050, 2000),
        voiced_ratio_range=(0.60, 0.85),
        purr_freq=(23, 28),
        signature="medium_meow",
        description="性格温和好奇，叫声中等清脆",
    ),

    # ============ 狗 (16 种) ============
    "金毛": BreedAcousticProfile(
        breed="金毛", pet_type="dog",
        f0_range=(280, 480),
        typical_duration=(0.20, 0.60),
        spectral_centroid_range=(900, 1800),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(8, 18),
        signature="soft_bark",
        description="中大型犬，温顺少吠，吠声柔和中等音高",
    ),
    "拉布拉多": BreedAcousticProfile(
        breed="拉布拉多", pet_type="dog",
        f0_range=(300, 500),
        typical_duration=(0.20, 0.60),
        spectral_centroid_range=(950, 1900),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(10, 20),
        signature="medium_bark",
        description="活泼友善，吠声响亮中等音高",
    ),
    "哈士奇": BreedAcousticProfile(
        breed="哈士奇", pet_type="dog",
        f0_range=(180, 520),
        typical_duration=(0.80, 3.50),
        spectral_centroid_range=(700, 1600),
        voiced_ratio_range=(0.80, 0.98),
        bark_rate=(2, 10),
        signature="howl_yodel",
        description="以嚎叫(howl)为主而非吠叫，长持续，带约德尔颤音",
    ),
    "柯基": BreedAcousticProfile(
        breed="柯基", pet_type="dog",
        f0_range=(420, 720),
        typical_duration=(0.15, 0.45),
        spectral_centroid_range=(1200, 2300),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(15, 30),
        signature="high_sharp_bark",
        description="小型犬，腿短共鸣腔小→F0偏高，吠声尖锐频繁",
    ),
    "泰迪": BreedAcousticProfile(
        breed="泰迪", pet_type="dog",
        f0_range=(450, 780),
        typical_duration=(0.15, 0.45),
        spectral_centroid_range=(1300, 2400),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(15, 35),
        signature="high_yappy_bark",
        description="玩具贵宾，体型小→F0高，爱叫，吠声尖锐",
    ),
    "边牧": BreedAcousticProfile(
        breed="边牧", pet_type="dog",
        f0_range=(320, 540),
        typical_duration=(0.20, 0.55),
        spectral_centroid_range=(1000, 2000),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(10, 25),
        signature="alert_bark",
        description="工作犬，警觉吠叫，音高中等",
    ),
    "德牧": BreedAcousticProfile(
        breed="德牧", pet_type="dog",
        f0_range=(200, 400),
        typical_duration=(0.25, 0.70),
        spectral_centroid_range=(700, 1500),
        voiced_ratio_range=(0.70, 0.92),
        bark_rate=(6, 15),
        signature="low_powerful_bark",
        description="大型工作犬，吠声低沉有力",
    ),
    "柴犬": BreedAcousticProfile(
        breed="柴犬", pet_type="dog",
        f0_range=(380, 620),
        typical_duration=(0.15, 0.45),
        spectral_centroid_range=(1100, 2100),
        voiced_ratio_range=(0.60, 0.88),
        bark_rate=(8, 18),
        signature="sharp_scream",
        description="安静少吠，但被惹时发出尖锐的柴犬尖叫",
    ),
    "萨摩耶": BreedAcousticProfile(
        breed="萨摩耶", pet_type="dog",
        f0_range=(280, 560),
        typical_duration=(0.30, 1.50),
        spectral_centroid_range=(900, 1800),
        voiced_ratio_range=(0.75, 0.95),
        bark_rate=(5, 12),
        signature="howl_talk",
        description="爱嚎叫和‘说话’，声音柔和带颤音",
    ),
    "阿拉斯加": BreedAcousticProfile(
        breed="阿拉斯加", pet_type="dog",
        f0_range=(150, 380),
        typical_duration=(0.80, 3.50),
        spectral_centroid_range=(600, 1300),
        voiced_ratio_range=(0.80, 0.98),
        bark_rate=(2, 8),
        signature="deep_howl",
        description="大型雪橇犬，胸腔大→F0极低，以低沉长嚎为主",
    ),
    "比熊": BreedAcousticProfile(
        breed="比熊", pet_type="dog",
        f0_range=(440, 760),
        typical_duration=(0.15, 0.45),
        spectral_centroid_range=(1300, 2400),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(15, 30),
        signature="high_bark",
        description="小型犬，吠声高而尖",
    ),
    "博美": BreedAcousticProfile(
        breed="博美", pet_type="dog",
        f0_range=(500, 850),
        typical_duration=(0.10, 0.40),
        spectral_centroid_range=(1500, 2700),
        voiced_ratio_range=(0.65, 0.92),
        bark_rate=(20, 40),
        signature="very_high_yappy",
        description="超小型犬，F0 最高，吠声极尖锐频繁",
    ),
    "法国斗牛犬": BreedAcousticProfile(
        breed="法国斗牛犬", pet_type="dog",
        f0_range=(220, 420),
        typical_duration=(0.20, 0.60),
        spectral_centroid_range=(700, 1500),
        voiced_ratio_range=(0.55, 0.80),
        bark_rate=(6, 15),
        signature="nasal_bark",
        description="短鼻犬，鼻腔共鸣→鼻音重，F0偏低，吠声沉闷",
    ),
    "雪纳瑞": BreedAcousticProfile(
        breed="雪纳瑞", pet_type="dog",
        f0_range=(360, 600),
        typical_duration=(0.20, 0.55),
        spectral_centroid_range=(1100, 2100),
        voiced_ratio_range=(0.65, 0.90),
        bark_rate=(12, 25),
        signature="alert_bark",
        description="警觉型梗犬，吠声响亮中等音高",
    ),
    "比格犬": BreedAcousticProfile(
        breed="比格犬", pet_type="dog",
        f0_range=(300, 520),
        typical_duration=(0.40, 1.50),
        spectral_centroid_range=(950, 1900),
        voiced_ratio_range=(0.75, 0.95),
        bark_rate=(15, 35),
        signature="bay_howl",
        description="嗅觉猎犬，发出独特的 bay 长嚎吠叫",
    ),
    "秋田犬": BreedAcousticProfile(
        breed="秋田犬", pet_type="dog",
        f0_range=(180, 380),
        typical_duration=(0.25, 0.80),
        spectral_centroid_range=(700, 1500),
        voiced_ratio_range=(0.70, 0.92),
        bark_rate=(4, 12),
        signature="low_bark",
        description="大型犬，安静，吠声低沉",
    ),
}


class BreedRecognizer:
    """
    基于声学特征的品种识别器
    输入: FeatureExtractor.extract_all_features() 的输出
    输出: Top-K 候选品种 + 匹配分数
    """

    def __init__(self, profiles: Dict[str, BreedAcousticProfile] = None):
        self.profiles = profiles or BREED_PROFILES

    def recognize(
        self,
        features: Dict,
        pet_type: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        识别品种
        Args:
            features: FeatureExtractor.extract_all_features() 的输出
                {
                  "spectral": {"centroid_mean", "duration", "rms", ...},
                  "pitch":    {"f0_mean", "f0_min", "f0_max", "voiced_ratio"}
                }
            pet_type: "cat" or "dog"
            top_k: 返回前 K 候选
        Returns:
            [{"breed": str, "score": float, "signature": str, "reason": str}, ...]
        """
        f0_mean = features.get("pitch", {}).get("f0_mean", 0) or 0
        duration = features.get("spectral", {}).get("duration", 0) or 0
        centroid = features.get("spectral", {}).get("centroid_mean", 0) or 0
        voiced = features.get("pitch", {}).get("voiced_ratio", 0) or 0

        scored = []
        for breed, prof in self.profiles.items():
            if prof.pet_type != pet_type:
                continue
            score, breakdown = self._score_breed(
                prof, f0_mean, duration, centroid, voiced
            )
            scored.append({
                "breed": breed,
                "score": round(score, 4),
                "signature": prof.signature,
                "description": prof.description,
                "breakdown": breakdown,
                "reason": self._explain(prof, breakdown),
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _score_breed(
        self,
        profile: BreedAcousticProfile,
        f0: float,
        dur: float,
        centroid: float,
        voiced: float,
    ) -> Tuple[float, Dict]:
        """加权打分 (0-1)"""
        f0_s = self._range_score(f0, profile.f0_range)
        dur_s = self._range_score(dur, profile.typical_duration)
        cen_s = self._range_score(centroid, profile.spectral_centroid_range)
        voi_s = self._range_score(voiced, profile.voiced_ratio_range)
        # 加权: F0 最关键(体型直接决定), 时长其次, 频谱质心反映喉部结构, 浊音比例辅助
        total = 0.40 * f0_s + 0.25 * dur_s + 0.20 * cen_s + 0.15 * voi_s
        return total, {
            "f0": round(f0_s, 3),
            "duration": round(dur_s, 3),
            "centroid": round(cen_s, 3),
            "voiced": round(voi_s, 3),
        }

    @staticmethod
    def _range_score(value: float, rng: Tuple[float, float]) -> float:
        """范围匹配度 (0-1), 落在区间内为 1, 距离越远分数越低"""
        if value <= 0:
            return 0.0
        lo, hi = rng
        if lo <= value <= hi:
            return 1.0
        if value < lo:
            return max(0.0, 1.0 - (lo - value) / max(lo, 1))
        return max(0.0, 1.0 - (value - hi) / max(hi, 1))

    @staticmethod
    def _explain(profile: BreedAcousticProfile, breakdown: Dict) -> str:
        """生成识别原因说明"""
        reasons = []
        if breakdown["f0"] >= 0.8:
            reasons.append(f"基频 {profile.f0_range} Hz 匹配(体型/共鸣腔)")
        if breakdown["duration"] >= 0.8:
            reasons.append(f"时长 {profile.typical_duration} s 匹配(发声习惯)")
        if breakdown["centroid"] >= 0.8:
            reasons.append(f"频谱质心 {profile.spectral_centroid_range} Hz 匹配(喉部结构)")
        if breakdown["voiced"] >= 0.8:
            reasons.append(f"浊音比例 {profile.voiced_ratio_range} 匹配")
        if profile.signature:
            reasons.append(f"特征签名: {profile.signature}")
        return "; ".join(reasons) if reasons else "特征部分匹配"


def list_breeds(pet_type: Optional[str] = None) -> List[str]:
    """列出所有支持的品种"""
    if pet_type:
        return [b for b, p in BREED_PROFILES.items() if p.pet_type == pet_type]
    return list(BREED_PROFILES.keys())


def get_profile(breed: str) -> Optional[BreedAcousticProfile]:
    """获取指定品种档案"""
    return BREED_PROFILES.get(breed)
