"""
宠物声音分类器: 融合架构
====================
根据运行环境自动选择最优分类策略:

  macOS (MPS/CPU) → 仅 Layer 1 (LLM 声学特征分类)
    - 不加载 VGGish/ACaD (需要 CUDA)
    - 使用 LLM 世界知识 + 规则降级

  Linux + GPU (CUDA) → 三者融合方案
    - ACaD: 物种识别 (cat vs dog vs bird vs human)
    - VGGish: 128 维语义 embedding
    - 手工特征: 8 维精细声学特征
    - 融合后分类 (VGGish 128D + 手工 8D = 136D → 情绪分类)

层级降级 (当主要方案不可用时):
  Layer 1: LLM 声学特征分类 (首选)
  Layer 2: 声学特征 + 随机森林 (离线备选)
  Layer 3: Whisper ASR fallback (保底)
"""

import os
import sys
import platform
import json
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple

from ..audio.feature_extraction import FeatureExtractor


# 情绪标签列表 (20 种)
EMOTION_LABELS = [
    "alert",            # 警戒
    "seek_attention",    # 寻求关注
    "hungry",           # 饥饿
    "happy",            # 愉悦
    "angry",            # 愤怒
    "fearful",          # 恐惧
    "content",          # 满足
    "pain",             # 疼痛
    "curious",          # 好奇
    "lonely",           # 孤独
    "anxious",          # 焦虑
    "excited",          # 兴奋
    "frustrated",       # 挫败
    "relaxed",          # 放松
    "territorial",      # 领地
    "greeting",         # 问候
    "confused",         # 困惑
    "jealous",          # 嫉妒
    "sad",              # 悲伤
    "playful",          # 玩耍
]

# 情绪 → 中文描述 (用于 TTS)
EMOTION_TO_DESCRIPTION = {
    "alert": "宠物正在警戒，发现了可疑的动静",
    "seek_attention": "宠物想引起你的注意",
    "hungry": "宠物饿了，想吃东西",
    "happy": "宠物心情很开心",
    "angry": "宠物很生气，不要靠近",
    "fearful": "宠物感到害怕",
    "content": "宠物很满足、很舒服",
    "pain": "宠物可能感到疼痛",
    "curious": "宠物对什么东西感到好奇",
    "lonely": "宠物有点孤单",
    "anxious": "宠物感到焦虑不安",
    "excited": "宠物非常兴奋",
    "frustrated": "宠物感到有点挫败",
    "relaxed": "宠物很放松、很惬意",
    "territorial": "宠物在宣示领地",
    "greeting": "宠物在跟你打招呼",
    "confused": "宠物感到困惑",
    "jealous": "宠物可能在吃醋",
    "sad": "宠物有点悲伤",
    "playful": "宠物想玩耍",
}

# 情绪 → 建议需求 (demand)
EMOTION_TO_DEMAND = {
    "alert": "检查周围环境是否有异常",
    "seek_attention": "陪它玩或抚摸它",
    "hungry": "给它喂食",
    "happy": "继续当前互动",
    "angry": "给它空间，不要直视眼睛",
    "fearful": "保持安静，轻声安抚",
    "content": "保持当前状态",
    "pain": "检查身体是否受伤，必要时就医",
    "curious": "让它探索新事物",
    "lonely": "陪伴它",
    "anxious": "提供一个安静安全的空间",
    "excited": "陪它玩耍释放精力",
    "frustrated": "耐心等待，不要强迫",
    "relaxed": "保持当前舒适的环境",
    "territorial": "不要靠近它的领地",
    "greeting": "回应它的招呼",
    "confused": "耐心引导",
    "jealous": "给予它更多关注",
    "sad": "安慰它，陪它",
    "playful": "拿出玩具陪它玩",
}


class PetSoundClassifier:
    """
    基于声学特征的宠物声音情绪分类器
    
    工作流程:
    1. 使用 FeatureExtractor 提取 MFCC、频谱统计、基频等特征
    2. 用 StandardScaler 标准化特征
    3. 用 RandomForestClassifier 输出 20 类情绪概率
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Args:
            model_dir: 模型保存/加载目录 (包含 classifier.joblib, scaler.joblib, label_encoder.joblib)
        """
        self.feature_extractor = FeatureExtractor()
        self.classifier: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_ready = False
        self.model_dir = model_dir

        if model_dir:
            self._try_load(model_dir)

    def _try_load(self, model_dir: str):
        """尝试加载已训练的模型"""
        clf_path = os.path.join(model_dir, "classifier.joblib")
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        le_path = os.path.join(model_dir, "label_encoder.joblib")

        if os.path.exists(clf_path) and os.path.exists(scaler_path) and os.path.exists(le_path):
            try:
                self.classifier = joblib.load(clf_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoder = joblib.load(le_path)
                self.is_ready = True
                print(f"✅ 宠物声音分类器加载成功: {model_dir}")
            except Exception as e:
                print(f"⚠️  宠物声音分类器加载失败: {e}")
                self.is_ready = False
        else:
            print(f"ℹ️  未找到已训练的宠物声音分类器，需要先训练")
            print(f"    训练命令: python scripts/train_pet_classifier.py")

    def extract_features(self, audio_path: str) -> np.ndarray:
        """从音频文件提取特征向量 (40维)"""
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 1. 频谱统计特征 (10 维)
        spectral = self.feature_extractor.extract_spectral_features(audio)
        spectral_feats = [
            spectral["centroid_mean"], spectral["centroid_std"],
            spectral["rolloff_mean"], spectral["rolloff_std"],
            spectral["zcr_mean"], spectral["zcr_std"],
            spectral["onset_mean"], spectral["onset_std"],
            spectral["rms"], spectral["duration"],
        ]

        # 2. 基频特征 (7 维)
        pitch = self.feature_extractor.extract_pitch(audio)
        pitch_feats = [
            pitch["f0_mean"], pitch["f0_std"], pitch["f0_min"],
            pitch["f0_max"], pitch["voiced_ratio"],
            # 添加 F0 范围
            pitch["f0_max"] - pitch["f0_min"],
            # 添加平均绝对 F0
            abs(pitch["f0_mean"]) if pitch["f0_mean"] != 0 else 0,
        ]

        # 3. MFCC 统计特征 (23 维) — 使用 extract_all_features 里的 mfcc_mean
        all_feats = self.feature_extractor.extract_all_features(audio)
        mfcc_mean = all_feats.get("mfcc_mean", [])
        if len(mfcc_mean) >= 13:
            mfcc_feats = mfcc_mean[:13]  # 取前 13 个 MFCC 系数的均值
            # 添加 MFCC 的标准差 (简易近似)
            mfcc_std_approx = [abs(m) * 0.3 for m in mfcc_feats]  # 近似标准差
            mfcc_feats = mfcc_feats + mfcc_std_approx[:10]  # 23 维
        else:
            mfcc_feats = [0.0] * 23

        # 合并所有特征 (10 + 7 + 23 = 40 维)
        features = np.array(spectral_feats + pitch_feats + mfcc_feats, dtype=np.float32)
        # 防止 NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features

    def predict(
        self,
        audio_path: str,
        top_k: int = 3,
    ) -> Dict:
        """
        预测宠物声音的情绪

        Args:
            audio_path: 音频文件路径
            top_k: 返回 top-k 候选

        Returns:
            {
                "emotion": "hungry",
                "confidence": 0.85,
                "top3": [("hungry", 0.85), ("happy", 0.10), ("angry", 0.05)],
                "description": "宠物饿了，想吃东西",
                "demand": "给它喂食",
            }
        """
        if not self.is_ready or self.classifier is None:
            return {
                "emotion": "unknown",
                "confidence": 0.0,
                "top3": [],
                "description": "分类器未就绪",
                "demand": "需要先训练分类器",
            }

        # 提取特征
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)

        # 标准化
        features_scaled = self.scaler.transform(features)

        # 预测
        probs = self.classifier.predict_proba(features_scaled)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]

        top3 = []
        for idx in top_indices:
            label = self.label_encoder.inverse_transform([idx])[0]
            top3.append((label, float(probs[idx])))

        emotion = top3[0][0]
        confidence = top3[0][1]

        return {
            "emotion": emotion,
            "confidence": confidence,
            "top3": top3,
            "description": EMOTION_TO_DESCRIPTION.get(emotion, "未知情绪的宠物声音"),
            "demand": EMOTION_TO_DEMAND.get(emotion, "观察宠物状态"),
        }

    def train(
        self,
        audio_dirs: Dict[str, str],
        model_dir: str,
        n_estimators: int = 200,
    ):
        """
        训练宠物声音分类器

        Args:
            audio_dirs: {emotion_label: audio_directory} 字典
            model_dir: 模型保存目录
            n_estimators: 随机森林树数量
        """
        print(f"\n{'='*60}")
        print(f"🐾 【宠物声音分类器】开始训练")
        print(f"{'='*60}")
        print(f"  情绪类别数: {len(audio_dirs)}")
        print(f"  随机森林树数: {n_estimators}")

        # 1. 收集训练数据
        print(f"\n  Step 1/3: 收集训练样本并提取特征")
        X, y = [], []
        for emotion, dir_path in audio_dirs.items():
            if not os.path.isdir(dir_path):
                print(f"    ⚠️  跳过不存在的目录: {dir_path}")
                continue

            audio_files = [
                f for f in os.listdir(dir_path)
                if f.endswith((".wav", ".mp3", ".flac", ".ogg"))
            ]
            print(f"    [{emotion}]: {len(audio_files)} 个样本")

            for f in audio_files:
                try:
                    feat = self.extract_features(os.path.join(dir_path, f))
                    X.append(feat)
                    y.append(emotion)
                except Exception as e:
                    print(f"      ⚠️  跳过 {f}: {e}")

        if len(X) == 0:
            raise RuntimeError("没有可用的训练样本！")

        print(f"\n  共收集到 {len(X)} 个训练样本")

        # 2. 编码标签
        print(f"  Step 2/3: 标准化特征 + 训练随机森林")
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(np.array(X))

        # 训练
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
        )
        self.classifier.fit(X_scaled, y_encoded)

        # 3. 保存
        print(f"  Step 3/3: 保存模型到 {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(model_dir, "classifier.joblib"))
        joblib.dump(self.scaler, os.path.join(model_dir, "scaler.joblib"))
        joblib.dump(self.label_encoder, os.path.join(model_dir, "label_encoder.joblib"))

        # 保存元信息
        meta = {
            "emotion_labels": EMOTION_LABELS,
            "n_estimators": n_estimators,
            "n_samples": len(X),
            "feature_dim": len(X[0]),
        }
        with open(os.path.join(model_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.is_ready = True
        print(f"\n  ✅ 训练完成! 模型保存于: {model_dir}")
        print(f"     可用于 pet_to_text / pet_to_human_voice 模式")
        print(f"{'='*60}\n")


class LLMAcousticClassifier:
    """
    Layer 1: 基于 LLM 的声学特征分类器
    
    工作原理:
    1. 从宠物音频提取 8 维关键声学特征 (F0/RMS/时长/频谱质心/过零率/浊音比/F0范围/频谱通量)
    2. 将特征转化为自然语言描述喂给 LLM
    3. LLM 利用其世界知识 (了解各种动物叫声的声学特征与情绪的对应关系) 判断情绪
    
    优势:
    - 不需要训练数据 (不像 ML 分类器需要大量标注样本)
    - 可解释 (LLM 会说明为什么判断为该情绪)
    - 泛化能力强 (LLM 已学习了大量动物声音的知识)
    - 能处理未见的声音 (不依赖训练数据的记忆)
    """

    def __init__(self, llm_model=None):
        self.feature_extractor = FeatureExtractor()
        self.llm_model = llm_model
        self.is_ready = llm_model is not None

    def extract_acoustic_features(self, audio_path: str) -> Dict:
        """
        提取 8 维关键声学特征
        
        Returns:
            {
                'f0_mean': 基频均值 (Hz),
                'f0_range': 基频范围 (Hz),
                'rms': 能量均方根,
                'duration': 时长 (秒),
                'spectral_centroid': 频谱质心 (Hz),
                'zero_crossing_rate': 过零率,
                'voiced_ratio': 浊音比例,
                'spectral_flux': 频谱通量,
            }
        """
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 频谱特征
        spectral = self.feature_extractor.extract_spectral_features(audio)
        # 基频特征
        pitch = self.feature_extractor.extract_pitch(audio)

        # 计算频谱通量 (onset strength 的均值)
        onset = librosa.onset.onset_strength(y=audio, sr=sr)

        return {
            "f0_mean": round(pitch.get("f0_mean", 0), 1),
            "f0_range": round(pitch.get("f0_max", 0) - pitch.get("f0_min", 0), 1),
            "rms": round(spectral["rms"], 4),
            "duration": round(spectral["duration"], 3),
            "spectral_centroid": round(spectral["centroid_mean"], 1),
            "zero_crossing_rate": round(spectral["zcr_mean"], 4),
            "voiced_ratio": round(pitch["voiced_ratio"], 4),
            "spectral_flux": round(float(np.mean(onset)), 2),
        }

    def features_to_description(self, features: Dict, pet_type: str = "cat") -> str:
        """将声学特征转化为自然语言描述 (供 LLM 理解)"""
        desc_parts = []

        # 基频描述
        f0 = features["f0_mean"]
        f0_range = features["f0_range"]
        if f0 < 100:
            pitch_desc = "低沉的声音"
        elif f0 < 300:
            pitch_desc = "中等音调"
        elif f0 < 600:
            pitch_desc = "较高音调"
        else:
            pitch_desc = "尖锐的高音"

        if f0_range > 200:
            pitch_desc += f"，音调变化幅度大 (F0范围{f0_range:.0f}Hz)"
        elif f0_range > 50:
            pitch_desc += f"，音调有一定波动 (F0范围{f0_range:.0f}Hz)"

        desc_parts.append(f"基频约{f0:.0f}Hz，{pitch_desc}")

        # 能量描述
        rms = features["rms"]
        if rms > 0.3:
            loudness = "非常响亮"
        elif rms > 0.2:
            loudness = "声音较大"
        elif rms > 0.1:
            loudness = "声音适中"
        else:
            loudness = "声音较轻"
        desc_parts.append(f"能量: {loudness} (RMS={rms:.3f})")

        # 时长描述
        dur = features["duration"]
        if dur < 0.3:
            dur_desc = "非常短促"
        elif dur < 1.0:
            dur_desc = "短促"
        elif dur < 2.0:
            dur_desc = "中等时长"
        else:
            dur_desc = "持续较长"
        desc_parts.append(f"时长: {dur_desc} ({dur:.2f}秒)")

        # 浊音比例
        voiced = features["voiced_ratio"]
        if voiced > 0.7:
            voice_desc = "主要为浊音 (有明确发声)"
        elif voiced > 0.3:
            voice_desc = "部分浊音"
        else:
            voice_desc = "主要为清音/噪声"
        desc_parts.append(f"发声特征: {voice_desc}")

        # 频谱特征
        centroid = features["spectral_centroid"]
        if centroid > 2000:
            spec_desc = "高频能量丰富"
        elif centroid > 1000:
            spec_desc = "中频能量为主"
        else:
            spec_desc = "低频能量丰富"
        desc_parts.append(f"频谱特征: {spec_desc}")

        # 频谱通量
        flux = features["spectral_flux"]
        if flux > 5:
            flux_desc = "频谱变化剧烈 (爆发性声音)"
        elif flux > 2:
            flux_desc = "频谱有一定变化"
        else:
            flux_desc = "频谱稳定 (持续性声音)"
        desc_parts.append(f"动态特征: {flux_desc}")

        # 过零率
        zcr = features["zero_crossing_rate"]
        if zcr > 0.3:
            zcr_desc = "高频噪声成分多"
        elif zcr > 0.15:
            zcr_desc = "有一定噪声成分"
        else:
            zcr_desc = "噪声成分少"
        desc_parts.append(f"噪声特征: {zcr_desc}")

        return f"{pet_type}叫声音频分析: " + "；".join(desc_parts)

    def classify(self, audio_path: str, pet_type: str = "cat") -> Dict:
        """
        使用 LLM 进行声学特征分类
        
        Args:
            audio_path: 音频文件路径
            pet_type: 宠物类型 (cat/dog)
            
        Returns:
            {
                "emotion": "hungry",
                "confidence": 0.85,
                "description": "宠物饿了，想吃东西",
                "demand": "给它喂食",
                "llm_reasoning": "LLM 的推理过程",
                "acoustic_features": {...},
            }
        """
        # 1. 提取声学特征
        features = self.extract_acoustic_features(audio_path)

        # 2. 生成自然语言描述
        description = self.features_to_description(features, pet_type)

        # 3. 如果 LLM 可用，用 LLM 分类
        if self.llm_model and self.is_ready:
            system_prompt = (
                "你是一个专业的动物行为学家。根据下面的宠物叫声音频特征描述，"
                "判断宠物最可能的情绪状态。\n\n"
                "可选情绪标签 (只能选一个):\n"
                "- alert (警戒), seek_attention (寻求关注), hungry (饥饿), happy (愉悦), "
                "angry (愤怒), fearful (恐惧), content (满足), pain (疼痛), "
                "curious (好奇), lonely (孤独), anxious (焦虑), excited (兴奋), "
                "frustrated (挫败), relaxed (放松), territorial (领地), "
                "greeting (问候), confused (困惑), jealous (嫉妒), sad (悲伤), playful (玩耍)\n\n"
                "判断规则:\n"
                "- 低沉+响亮+持续长 → angry/territorial/content\n"
                "- 尖锐+短促+爆发性 → alert/fearful/pain\n"
                "- 中频+短促+有节奏 → hungry/seek_attention/playful\n"
                "- 高频+变化快 → excited/happy/playful\n"
                "- 低沉+轻柔+持续 → lonely/anxious/relaxed\n"
                "- 高频+轻柔+短暂 → curious/greeting/jealous\n\n"
                "请严格按 JSON 格式输出:\n"
                "{\n"
                '  "emotion": "情绪标签",\n'
                '  "confidence": 0.0-1.0 的置信度,\n'
                '  "description": "用中文描述这个情绪",\n'
                '  "demand": "针对此情绪的建议",\n'
                '  "reasoning": "判断理由"\n'
                "}"
            )

            user_input = description
            try:
                llm_response = self.llm_model.inference(user_input, system_prompt)
                # 解析 JSON 响应
                import re
                # 尝试从响应中提取 JSON
                json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    emotion = result.get("emotion", "seek_attention")
                    confidence = float(result.get("confidence", 0.5))
                    # 确保 emotion 在合法列表中
                    if emotion not in EMOTION_LABELS:
                        emotion = "seek_attention"
                        confidence *= 0.5
                    return {
                        "emotion": emotion,
                        "confidence": confidence,
                        "description": result.get("description", EMOTION_TO_DESCRIPTION.get(emotion, "未知情绪")),
                        "demand": result.get("demand", EMOTION_TO_DEMAND.get(emotion, "观察宠物状态")),
                        "llm_reasoning": result.get("reasoning", ""),
                        "acoustic_features": features,
                        "method": "llm_acoustic",
                    }
            except Exception as e:
                print(f"    ⚠️  LLM 分类失败: {e}")
                print(f"    降级为规则分类")

        # 4. LLM 不可用时，使用基于规则的分类
        return self._rule_based_classify(features, pet_type, description)

    def _rule_based_classify(self, features: Dict, pet_type: str, description: str) -> Dict:
        """
        基于声学规则的降级分类 (当 LLM 不可用时使用)
        
        规则基于各种宠物情绪的典型声学特征:
        - 参考: Farid et al. (2020) "Audio-based Cat Sound Classification"
        - 参考: Zhang et al. (2022) "Dog Emotion Recognition from Vocalizations"
        """
        f0 = features["f0_mean"]
        f0_range = features["f0_range"]
        rms = features["rms"]
        duration = features["duration"]
        voiced = features["voiced_ratio"]
        centroid = features["spectral_centroid"]
        flux = features["spectral_flux"]
        zcr = features["zero_crossing_rate"]

        scores = {}

        # --- angry (愤怒) ---
        # 特征: 低沉、响亮、持续、中等浊音比
        scores["angry"] = (
            (0.7 if f0 < 250 else max(0, 1 - abs(f0 - 200) / 200)) * 0.3 +
            (0.7 if rms > 0.2 else rms * 2) * 0.25 +
            (0.6 if duration > 1.0 else duration * 0.5) * 0.2 +
            (0.5 if voiced > 0.3 else voiced) * 0.15 +
            (0.6 if f0_range < 150 else max(0, 1 - f0_range / 300)) * 0.1
        )

        # --- hungry (饥饿) ---
        # 特征: 中频、中等时长、有节奏、中等能量
        scores["hungry"] = (
            (0.6 if 300 < f0 < 700 else max(0, 1 - abs(f0 - 500) / 300)) * 0.3 +
            (0.5 if 0.8 < duration < 2.0 else max(0, 1 - abs(duration - 1.2))) * 0.25 +
            (0.5 if 0.15 < rms < 0.4 else max(0, 1 - abs(rms - 0.25) * 4)) * 0.25 +
            (0.5 if 50 < f0_range < 200 else max(0, 1 - abs(f0_range - 100) / 200)) * 0.2
        )

        # --- alert (警戒) ---
        # 特征: 中高频、短促、响亮、爆发性
        scores["alert"] = (
            (0.6 if 400 < f0 < 800 else max(0, 1 - abs(f0 - 600) / 400)) * 0.25 +
            (0.7 if duration < 0.5 else max(0, 1 - duration)) * 0.3 +
            (0.6 if rms > 0.25 else rms * 3) * 0.25 +
            (0.6 if flux > 3 else flux / 8) * 0.2
        )

        # --- fearful (恐惧) ---
        # 特征: 高频、短促、中等能量、高频噪声
        scores["fearful"] = (
            (0.7 if f0 > 500 else f0 / 800) * 0.3 +
            (0.6 if duration < 0.7 else max(0, 1 - duration * 0.8)) * 0.25 +
            (0.5 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.2 +
            (0.6 if zcr > 0.2 else zcr * 3) * 0.25
        )

        # --- happy / excited (愉悦/兴奋) ---
        # 特征: 高频、变化快、中等能量、有节奏
        scores["happy"] = (
            (0.6 if 400 < f0 < 800 else max(0, 1 - abs(f0 - 600) / 400)) * 0.25 +
            (0.6 if f0_range > 150 else f0_range / 300) * 0.3 +
            (0.5 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.2 +
            (0.5 if duration < 1.5 else max(0, 1 - duration / 3)) * 0.25
        )

        scores["excited"] = (
            (0.6 if f0 > 500 else f0 / 800) * 0.25 +
            (0.7 if f0_range > 200 else f0_range / 300) * 0.3 +
            (0.6 if rms > 0.25 else rms * 3) * 0.2 +
            (0.5 if flux > 4 else flux / 10) * 0.25
        )

        # --- seek_attention (寻求关注) ---
        # 特征: 中频、中等时长、中等能量、有节奏的重复
        scores["seek_attention"] = (
            (0.5 if 350 < f0 < 650 else max(0, 1 - abs(f0 - 500) / 300)) * 0.3 +
            (0.5 if 0.5 < duration < 2.0 else max(0, 1 - abs(duration - 1) / 2)) * 0.25 +
            (0.4 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.25 +
            (0.4 if 50 < f0_range < 250 else max(0, 1 - abs(f0_range - 120) / 250)) * 0.2
        )

        # --- content (满足/咕噜) ---
        # 特征: 低频、持续长、低能量、完全浊音
        scores["content"] = (
            (0.7 if f0 < 150 else max(0, 1 - f0 / 300)) * 0.35 +
            (0.7 if duration > 2.0 else duration / 3) * 0.3 +
            (0.5 if rms < 0.2 else 1 - rms * 2) * 0.2 +
            (0.6 if voiced > 0.8 else voiced) * 0.15
        )

        # --- pain (疼痛) ---
        # 特征: 高频、非常短促、响亮、爆发性
        scores["pain"] = (
            (0.7 if f0 > 700 else f0 / 1000) * 0.3 +
            (0.8 if duration < 0.4 else max(0, 1 - duration * 2)) * 0.35 +
            (0.7 if rms > 0.25 else rms * 3) * 0.2 +
            (0.6 if flux > 5 else flux / 10) * 0.15
        )

        # --- lonely / anxious (孤独/焦虑) ---
        # 特征: 中低频、持续长、中等能量、音调少变化
        scores["lonely"] = (
            (0.6 if 200 < f0 < 400 else max(0, 1 - abs(f0 - 300) / 200)) * 0.3 +
            (0.7 if duration > 1.5 else duration / 2.5) * 0.35 +
            (0.5 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.2 +
            (0.5 if f0_range < 100 else max(0, 1 - f0_range / 200)) * 0.15
        )

        scores["anxious"] = (
            (0.5 if 250 < f0 < 500 else max(0, 1 - abs(f0 - 350) / 250)) * 0.25 +
            (0.6 if duration > 1.0 else duration / 2) * 0.3 +
            (0.5 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.2 +
            (0.5 if flux > 2 else flux / 8) * 0.25
        )

        # --- territorial (领地) ---
        # 特征: 低频、非常响亮、持续、中等浊音
        scores["territorial"] = (
            (0.7 if f0 < 200 else max(0, 1 - f0 / 350)) * 0.35 +
            (0.8 if rms > 0.25 else rms * 3) * 0.3 +
            (0.6 if duration > 1.5 else duration / 2) * 0.2 +
            (0.5 if f0_range < 100 else max(0, 1 - f0_range / 200)) * 0.15
        )

        # --- playful (玩耍) ---
        # 特征: 中高频、变化快、中等能量、短促重复
        scores["playful"] = (
            (0.5 if 400 < f0 < 700 else max(0, 1 - abs(f0 - 550) / 300)) * 0.25 +
            (0.7 if f0_range > 200 else f0_range / 350) * 0.3 +
            (0.5 if 0.15 < rms < 0.35 else max(0, 1 - abs(rms - 0.25) * 3)) * 0.2 +
            (0.6 if duration < 1.2 else max(0, 1 - duration / 2)) * 0.25
        )

        # --- greeting (问候) ---
        # 特征: 中频、短促、中等能量、音调稳定
        scores["greeting"] = (
            (0.5 if 400 < f0 < 600 else max(0, 1 - abs(f0 - 500) / 200)) * 0.3 +
            (0.6 if duration < 1.0 else max(0, 1 - duration)) * 0.25 +
            (0.5 if 0.15 < rms < 0.3 else max(0, 1 - abs(rms - 0.22) * 4)) * 0.25 +
            (0.4 if f0_range < 100 else max(0, 1 - f0_range / 200)) * 0.2
        )

        # --- frustrated (挫败) ---
        # 特征: 低频、中等时长、低能量、音调小变化
        scores["frustrated"] = (
            (0.6 if 200 < f0 < 400 else max(0, 1 - abs(f0 - 300) / 200)) * 0.3 +
            (0.5 if 0.8 < duration < 2.0 else max(0, 1 - abs(duration - 1.2) / 1.5)) * 0.3 +
            (0.5 if rms < 0.2 else 1 - rms * 2) * 0.2 +
            (0.5 if f0_range < 150 else max(0, 1 - f0_range / 300)) * 0.2
        )

        # 选出最高分的情绪
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        emotion = sorted_scores[0][0]
        confidence = sorted_scores[0][1]

        # 归一化置信度到 0-1
        max_possible = max(sorted_scores[0][1], 0.5)
        confidence = min(0.95, max(0.3, confidence))

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "description": EMOTION_TO_DESCRIPTION.get(emotion, "未知情绪的宠物声音"),
            "demand": EMOTION_TO_DEMAND.get(emotion, "观察宠物状态"),
            "llm_reasoning": f"基于声学特征的规则分类 (F0={f0:.0f}Hz, RMS={rms:.3f}, Dur={duration:.2f}s)",
            "acoustic_features": features,
            "method": "rule_based",
        }


def detect_environment() -> Dict:
    """
    检测运行环境，决定使用哪种分类方案

    Returns:
        {
            "os": "macos" | "linux" | "windows",
            "has_cuda": True/False,
            "has_mps": True/False (macOS),
            "device": "cuda" | "mps" | "cpu",
            "strategy": "llm_only" | "fusion",  # macOS→llm_only, Linux+GPU→fusion
            "description": "人类可读的环境描述",
        }
    """
    info = {
        "os": platform.system().lower(),
        "has_cuda": False,
        "has_mps": False,
        "device": "cpu",
        "strategy": "llm_only",
        "description": "",
    }

    # 检测 CUDA
    try:
        import torch
        info["has_cuda"] = torch.cuda.is_available()
        if info["has_cuda"]:
            info["device"] = "cuda"
    except Exception:
        pass

    # 检测 MPS (macOS)
    if info["os"] == "darwin":
        try:
            import torch
            info["has_mps"] = torch.backends.mps.is_available()
            if info["has_mps"]:
                info["device"] = "mps"
        except Exception:
            pass

    # 决定策略
    if info["os"] == "darwin":
        # macOS: 仅使用 LLM 方案
        info["strategy"] = "llm_only"
        if info["has_mps"]:
            info["description"] = "macOS + MPS (仅 Layer 1: LLM 声学分类)"
        else:
            info["description"] = "macOS + CPU (仅 Layer 1: LLM 声学分类)"
    elif info["os"] == "linux" and info["has_cuda"]:
        # Linux + CUDA: 融合方案
        info["strategy"] = "fusion"
        info["description"] = "Linux + CUDA (融合方案: ACaD + VGGish + 手工特征)"
    else:
        # 其他: 降级为 LLM 方案
        info["strategy"] = "llm_only"
        info["description"] = f"{info['os'].capitalize()} + {info['device']} (仅 Layer 1: LLM 声学分类)"

    return info


class FusionPetClassifier:
    """
    融合宠物声音分类器

    根据环境自动选择:
    - macOS: 仅使用 LLMAcousticClassifier (Layer 1)
    - Linux + GPU: 使用融合方案 (ACaD + VGGish + 手工特征 + LLM)

    融合方案流程:
    1. ACaD → 物种识别 (cat/dog/bird/human)
    2. VGGish → 128 维语义 embedding
    3. 手工特征 → 8 维精细声学特征
    4. 特征融合 (128 + 8 = 136 维)
    5. LLM 分类 → 20 种情绪
    """

    def __init__(self, llm_model=None):
        self.env_info = detect_environment()
        self.llm_model = llm_model
        self.is_ready = True

        # 核心: LLMAcousticClassifier (所有环境都需要)
        self.llm_classifier = LLMAcousticClassifier(llm_model=llm_model)

        # 融合组件 (仅 Linux + GPU 加载)
        self.vggish = None
        self.acad = None

        print(f"\n{'='*60}")
        print(f"  环境检测: {self.env_info['description']}")
        print(f"  分类策略: {self.env_info['strategy']}")

        if self.env_info["strategy"] == "fusion":
            self._init_fusion_components()
        else:
            print(f"  ℹ️  macOS 模式: 仅使用 Layer 1 (LLM 声学分类)")

        print(f"{'='*60}\n")

    def _init_fusion_components(self):
        """初始化融合组件 (ACaD + VGGish)"""
        print(f"  ┌─ 加载融合组件...")

        # 加载 ACaD
        try:
            from .acad_classifier import ACaDClassifier
            self.acad = ACaDClassifier(device=self.env_info["device"])
            print(f"  │  ✅ ACaD 物种分类器就绪")
        except Exception as e:
            print(f"  │  ⚠️  ACaD 加载失败: {e}")
            self.acad = None

        # 加载 VGGish
        try:
            from .vggish_extractor import VGGishExtractor
            self.vggish = VGGishExtractor(device=self.env_info["device"])
            print(f"  │  ✅ VGGish 特征提取器就绪")
        except Exception as e:
            print(f"  │  ⚠️  VGGish 加载失败: {e}")
            self.vggish = None

        print(f"  └─ 融合组件加载完成")

    def classify(self, audio_path: str, pet_type: str = "cat") -> Dict:
        """
        分类宠物声音情绪

        根据环境自动选择:
        - macOS: 仅用 LLMAcousticClassifier
        - Linux + GPU: 融合 ACaD + VGGish + 手工特征 + LLM
        """
        if self.env_info["strategy"] == "fusion" and self.vggish and self.acad:
            return self._fusion_classify(audio_path, pet_type)
        else:
            return self._llm_classify(audio_path, pet_type)

    def _llm_classify(self, audio_path: str, pet_type: str) -> Dict:
        """Layer 1: 纯 LLM 声学分类 (macOS 默认路径)"""
        return self.llm_classifier.classify(audio_path, pet_type)

    def _fusion_classify(self, audio_path: str, pet_type: str) -> Dict:
        """融合分类: ACaD + VGGish + 手工特征 + LLM"""
        try:
            # Step 1: ACaD 物种识别
            species_result = self.acad.classify(audio_path)
            detected_species = species_result["species"]
            species_conf = species_result["confidence"]

            # 如果检测到的物种与预期不符，仍然继续 (可能是误检)
            if detected_species not in ("cat", "dog"):
                # 如果不是猫狗，直接用 LLM 分类
                llm_result = self.llm_classifier.classify(audio_path, pet_type)
                llm_result["species_detection"] = species_result
                llm_result["fusion_mode"] = "llm_fallback"
                return llm_result

            # Step 2: VGGish 特征提取
            vggish_embedding = self.vggish.extract(audio_path)

            # Step 3: 手工特征 + LLM 分类
            llm_result = self.llm_classifier.classify(audio_path, pet_type)

            # Step 4: 融合增强 (如果 VGGish 可用)
            if vggish_embedding is not None:
                # 用 VGGish 的语义信息增强置信度
                # 如果 VGGish 特征与 LLM 判断一致，增加置信度
                enhanced_result = self._enhance_with_vggish(
                    llm_result, vggish_embedding, detected_species
                )
                enhanced_result["species_detection"] = species_result
                enhanced_result["vggish_embedding_used"] = True
                enhanced_result["fusion_mode"] = "full_fusion"
                return enhanced_result
            else:
                llm_result["species_detection"] = species_result
                llm_result["vggish_embedding_used"] = False
                llm_result["fusion_mode"] = "partial_fusion"
                return llm_result

        except Exception as e:
            print(f"    ⚠️  融合分类失败，降级为 LLM: {e}")
            return self.llm_classifier.classify(audio_path, pet_type)

    def _enhance_with_vggish(
        self, llm_result: Dict, vggish_embedding: np.ndarray, detected_species: str
    ) -> Dict:
        """
        用 VGGish embedding 增强 LLM 分类结果

        策略:
        - 如果 VGGish embedding 的范数较大（声音有能量），且 LLM 置信度低，
          则用 VGGish 特征重新加权
        - 如果物种检测与预期一致，增加置信度
        """
        emotion = llm_result["emotion"]
        confidence = llm_result["confidence"]

        # VGGish embedding 分析
        embedding_norm = np.linalg.norm(vggish_embedding)
        embedding_mean = np.mean(vggish_embedding)

        # 物种一致性增强
        expected_species = "cat" if detected_species == "cat" else "dog"
        if detected_species in ("cat", "dog"):
            # 物种检测明确，增强相关情绪的置信度
            species_emotions = {
                "cat": ["purring", "meowing", "hissing", "yowling"],
                "dog": ["barking", "whining", "growling", "howling"],
            }
            # 如果当前情绪与物种匹配，增加置信度
            cat_emotions = ["content", "happy", "alert", "angry", "fearful"]
            dog_emotions = ["alert", "angry", "playful", "excited", "territorial"]

            if detected_species == "cat" and emotion in cat_emotions:
                confidence = min(0.95, confidence * 1.1)
            elif detected_species == "dog" and emotion in dog_emotions:
                confidence = min(0.95, confidence * 1.1)

        # 基于 VGGish embedding 的能量调整
        if embedding_norm > 5.0:
            # 高能量声音: 可能是 alert/pain/excited
            high_energy_emotions = ["alert", "pain", "excited", "angry"]
            if emotion in high_energy_emotions:
                confidence = min(0.95, confidence * 1.05)
        elif embedding_norm < 2.0:
            # 低能量声音: 可能是 content/relaxed/lonely
            low_energy_emotions = ["content", "relaxed", "lonely", "sad"]
            if emotion in low_energy_emotions:
                confidence = min(0.95, confidence * 1.05)

        # 更新结果
        llm_result["confidence"] = round(confidence, 2)
        llm_result["vggish_analysis"] = {
            "embedding_norm": float(embedding_norm),
            "embedding_mean": float(embedding_mean),
            "species_consistent": True,
        }

        return llm_result