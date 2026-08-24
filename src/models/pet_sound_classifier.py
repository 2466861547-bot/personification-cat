"""
宠物声音分类器: 基于声学特征 + 随机森林 (Random Forest)
用于替代 Whisper 处理宠物声音 (非 ASR 任务，而是音频分类任务)

支持 20 种情绪标签:
- alert, seek_attention, hungry, happy, angry, fearful, content, pain
- curious, lonely, anxious, excited, frustrated, relaxed, territorial
- greeting, confused, jealous, sad, playful
"""

import os
import json
import numpy as np
import joblib
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

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