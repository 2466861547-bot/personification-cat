"""
ACaD (Audio Cats and Dogs) 物种分类器
======================================
基于预训练 CNN 模型识别音频中的物种 (cat vs dog)。
也支持其他物种分类 (bird, human 等)。

参考: "Audio Classification of Cats and Dogs" (开源实现)
依赖: torch, torchaudio
"""

import os
import sys
import numpy as np
from typing import Dict, Optional

import torch
import torch.nn as nn


class ACaDClassifier:
    """
    ACaD 物种分类器

    使用轻量级 1D CNN 识别音频物种:
    - 输入: 原始音频波形 (16kHz)
    - 输出: 物种概率分布 (cat/dog/bird/human/other)

    在没有预训练权重时，使用基于特征的规则分类作为降级方案
    """

    # 支持的物种标签
    SPECIES_LABELS = ["cat", "dog", "bird", "human", "other"]

    # 各物种的典型声学特征范围 (基于文献统计)
    SPECIES_PROFILES = {
        "cat": {
            "f0_range": (500, 2000),       # 猫叫基频 500-2000Hz
            "rms_range": (0.15, 0.35),     # 中等能量
            "duration_range": (0.3, 2.0),  # 短促到中等时长
            "centroid_range": (1000, 3000),  # 中频为主
            "zcr_range": (0.05, 0.20),     # 适中过零率
        },
        "dog": {
            "f0_range": (100, 800),        # 狗吠基频 100-800Hz
            "rms_range": (0.15, 0.45),     # 能量范围广
            "duration_range": (0.2, 1.5),  # 短促
            "centroid_range": (500, 2000),  # 低频为主
            "zcr_range": (0.03, 0.15),     # 低过零率
        },
        "bird": {
            "f0_range": (2000, 5000),      # 鸟鸣基频高
            "rms_range": (0.10, 0.30),
            "duration_range": (0.1, 0.8),  # 非常短促
            "centroid_range": (2000, 5000),  # 高频为主
            "zcr_range": (0.15, 0.40),     # 高过零率
        },
        "human": {
            "f0_range": (80, 400),         # 人类语音基频
            "rms_range": (0.10, 0.40),
            "duration_range": (0.5, 3.0),
            "centroid_range": (500, 3000),
            "zcr_range": (0.03, 0.12),
        },
    }

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """加载 ACaD 分类模型"""
        try:
            # 尝试加载预训练模型
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "checkpoints", "acad", "model.pt"
            )

            if os.path.exists(model_path):
                # 加载预训练权重
                model = self._build_model(num_classes=len(self.SPECIES_LABELS))
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                self.model = model.to(self.device)
                self.model.eval()
                self.is_ready = True
                print(f"    ✅ ACaD 预训练模型加载成功: {model_path}")
            else:
                # 没有预训练模型，使用规则分类 (仍然可用)
                self.model = None
                self.is_ready = True
                print(f"    ℹ️  ACaD 预训练模型不存在，使用基于特征的物种识别")

        except Exception as e:
            print(f"    ⚠️  ACaD 模型加载失败: {e}")
            self.model = None
            self.is_ready = True  # 降级为规则模式

    def _build_model(self, num_classes: int = 5) -> nn.Module:
        """构建轻量级 1D CNN 分类器"""
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x

        return SimpleCNN(num_classes)

    def classify(self, audio_path: str) -> Dict:
        """
        分类音频物种

        Args:
            audio_path: 音频文件路径

        Returns:
            {
                "species": "cat",
                "confidence": 0.85,
                "scores": {"cat": 0.85, "dog": 0.10, ...},
                "method": "cnn" | "rule_based"
            }
        """
        if self.model is not None:
            return self._cnn_classify(audio_path)
        else:
            return self._rule_based_classify(audio_path)

    def _cnn_classify(self, audio_path: str) -> Dict:
        """使用 CNN 模型分类"""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # 重采样到 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)

            # 单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # 截断或填充到固定长度 (2 秒)
            target_len = 32000  # 2秒 @ 16kHz
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            elif waveform.shape[1] < target_len:
                padding = torch.zeros(1, target_len - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)

            waveform = waveform.to(self.device)

            with torch.no_grad():
                logits = self.model(waveform)
                probs = torch.softmax(logits, dim=-1)
                conf, pred_idx = probs.max(dim=-1)

            species = self.SPECIES_LABELS[pred_idx.item()]
            confidence = conf.item()

            scores = {
                self.SPECIES_LABELS[i]: probs[0, i].item()
                for i in range(len(self.SPECIES_LABELS))
            }

            return {
                "species": species,
                "confidence": confidence,
                "scores": scores,
                "method": "cnn",
            }

        except Exception as e:
            print(f"    ⚠️  ACaD CNN 分类失败: {e}")
            return self._rule_based_classify(audio_path)

    def _rule_based_classify(self, audio_path: str) -> Dict:
        """
        基于声学特征的规则分类 (降级方案)

        分析音频的 F0、能量、频谱质心等特征，与各物种的典型特征范围匹配
        """
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

            # 提取声学特征
            f0 = self._estimate_f0(audio, sr)
            rms = np.sqrt(np.mean(audio ** 2))
            duration = len(audio) / sr
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
            zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()

            # 计算每个物种的匹配分数
            scores = {}
            for species, profile in self.SPECIES_PROFILES.items():
                score = self._compute_match_score(
                    f0, rms, duration, centroid, zcr, profile
                )
                scores[species] = score

            # 归一化
            total = sum(scores.values()) + 1e-8
            scores_norm = {k: v / total for k, v in scores.items()}

            # 选出最高分
            species = max(scores_norm, key=scores_norm.get)
            confidence = scores_norm[species]

            return {
                "species": species,
                "confidence": confidence,
                "scores": scores_norm,
                "method": "rule_based",
            }

        except Exception as e:
            print(f"    ⚠️  ACaD 规则分类失败: {e}")
            return {
                "species": "unknown",
                "confidence": 0.0,
                "scores": {s: 0.2 for s in self.SPECIES_LABELS},
                "method": "failed",
            }

    def _estimate_f0(self, audio: np.ndarray, sr: int) -> float:
        """估计基频 (使用自相关法简化版)"""
        try:
            # 使用 librosa 的 pyin 算法
            f0s = librosa.pyin(audio, fmin=50, fmax=4000, sr=sr)
            valid_f0s = f0s[f0s > 0]
            if len(valid_f0s) > 0:
                return float(np.median(valid_f0s))
        except Exception:
            pass
        return 0.0

    def _compute_match_score(
        self,
        f0: float,
        rms: float,
        duration: float,
        centroid: float,
        zcr: float,
        profile: Dict,
    ) -> float:
        """计算特征与物种画像的匹配分数"""
        score = 1.0

        # F0 匹配
        f0_min, f0_max = profile["f0_range"]
        if f0_min <= f0 <= f0_max:
            score *= 1.5  # 命中范围加分
        elif f0 > 0:
            # 距离越近分数越高
            dist = min(abs(f0 - f0_min), abs(f0 - f0_max))
            score *= max(0.1, 1.0 - dist / 1000)

        # RMS 匹配
        rms_min, rms_max = profile["rms_range"]
        if rms_min <= rms <= rms_max:
            score *= 1.3
        else:
            dist = min(abs(rms - rms_min), abs(rms - rms_max))
            score *= max(0.3, 1.0 - dist * 5)

        # 时长匹配
        dur_min, dur_max = profile["duration_range"]
        if dur_min <= duration <= dur_max:
            score *= 1.2
        else:
            dist = min(abs(duration - dur_min), abs(duration - dur_max))
            score *= max(0.5, 1.0 - dist / 5)

        # 频谱质心匹配
        cent_min, cent_max = profile["centroid_range"]
        if cent_min <= centroid <= cent_max:
            score *= 1.3
        elif centroid > 0:
            dist = min(abs(centroid - cent_min), abs(centroid - cent_max))
            score *= max(0.1, 1.0 - dist / 3000)

        return score