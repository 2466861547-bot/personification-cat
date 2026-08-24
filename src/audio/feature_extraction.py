"""
音频特征提取: Mel频谱、MFCC、声学特征
"""

import librosa
import numpy as np
import torch
from typing import Dict, Optional


class FeatureExtractor:
    """提取音频声学特征"""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 512,
        f_min: float = 50,
        f_max: float = 8000,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """提取 Mel 频谱图 (Whisper 微调用)"""
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """提取 MFCC 特征"""
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            # delta 和 delta-delta (对短音频可能失败)
            try:
                delta = librosa.feature.delta(mfcc)
            except Exception:
                delta = np.zeros_like(mfcc)
            try:
                delta2 = librosa.feature.delta(mfcc, order=2)
            except Exception:
                delta2 = np.zeros_like(mfcc)
            return np.concatenate([mfcc, delta, delta2], axis=0)
        except Exception:
            # 极短音频返回全零特征
            return np.zeros((n_mfcc * 3, 1), dtype=np.float32)

    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """提取频谱统计特征 (用于声音分类)"""
        # 质心频率
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        # 频谱滚降点
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        # 过零率
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=self.hop_length)
        # 频谱通量
        onset = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )

        return {
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "rolloff_mean": float(np.mean(rolloff)),
            "rolloff_std": float(np.std(rolloff)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "onset_mean": float(np.mean(onset)),
            "onset_std": float(np.std(onset)),
            "rms": float(np.sqrt(np.mean(audio**2))),
            "duration": float(len(audio) / self.sample_rate),
        }

    def extract_pitch(self, audio: np.ndarray) -> Dict[str, float]:
        """提取基频(F0)特征"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sample_rate,
            )
        except Exception:
            # pyin 对极短音频可能报错，返回零特征
            return {
                "f0_mean": 0.0, "f0_std": 0.0, "f0_min": 0.0,
                "f0_max": 0.0, "voiced_ratio": 0.0,
            }
        f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        return {
            "f0_mean": float(np.mean(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "f0_std": float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "f0_min": float(np.min(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "f0_max": float(np.max(f0_clean)) if len(f0_clean) > 0 else 0.0,
            "voiced_ratio": float(np.mean(voiced_flag)) if f0 is not None else 0.0,
        }

    def extract_all_features(self, audio: np.ndarray) -> Dict:
        """提取全部特征"""
        return {
            "spectral": self.extract_spectral_features(audio),
            "pitch": self.extract_pitch(audio),
            "mfcc_mean": self.extract_mfcc(audio).mean(axis=1).tolist(),
        }

    def to_log_mel_tensor(
        self, audio: np.ndarray, max_length: int = 3000
    ) -> torch.Tensor:
        """转换为 Whisper 格式的 log-Mel 张量"""
        mel = self.extract_mel_spectrogram(audio)
        # 转置: (time, n_mels)
        mel = mel.T
        # 截断或填充
        if mel.shape[0] < max_length:
            pad = np.zeros((max_length - mel.shape[0], self.n_mels))
            mel = np.concatenate([mel, pad], axis=0)
        else:
            mel = mel[:max_length]

        return torch.from_numpy(mel).float()
