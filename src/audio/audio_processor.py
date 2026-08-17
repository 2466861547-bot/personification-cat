"""
音频处理器: 加载、分段、标准化音频
"""

import os
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class AudioSegment:
    """音频段"""
    audio: np.ndarray
    sample_rate: int
    start_time: float
    duration: float
    file_path: str = ""


class AudioProcessor:
    """音频加载、分段、标准化处理器"""

    def __init__(
        self,
        sample_rate: int = 16000,
        mono: bool = True,
        segment_duration: float = 5.0,
        hop_duration: float = 2.0,
    ):
        self.sample_rate = sample_rate
        self.mono = mono
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.segment_samples = int(segment_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件并标准化"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")

        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=self.mono,
        )
        return audio, sr

    def normalize_audio(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """音量标准化到目标分贝"""
        rms = np.sqrt(np.mean(audio**2) + 1e-10)
        target_rms = 10 ** (target_db / 20)
        gain = target_rms / (rms + 1e-10)
        return audio * gain

    def segment_audio(self, audio: np.ndarray, sr: int) -> List[AudioSegment]:
        """滑动窗口分段"""
        segments = []
        total_samples = len(audio)

        if total_samples < self.segment_samples:
            # 音频短于一段,补零
            padded = np.zeros(self.segment_samples, dtype=np.float32)
            padded[:total_samples] = audio
            segments.append(AudioSegment(
                audio=padded,
                sample_rate=sr,
                start_time=0.0,
                duration=self.segment_duration,
            ))
            return segments

        for start in range(0, total_samples - self.segment_samples + 1, self.hop_samples):
            segment_audio = audio[start : start + self.segment_samples]
            segments.append(AudioSegment(
                audio=segment_audio,
                sample_rate=sr,
                start_time=start / sr,
                duration=self.segment_duration,
            ))

        return segments

    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None):
        """保存音频文件"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sr = sr or self.sample_rate
        sf.write(file_path, audio, sr)

    def augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """数据增强: 加噪、变速、音调偏移"""
        try:
            from audiomentations import (
                Compose, AddGaussianNoise, TimeStretch, PitchShift
            )
            augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            ])
            return augment(samples=audio, sample_rate=sr)
        except ImportError:
            return audio

    def extract_clips_from_long_audio(
        self,
        file_path: str,
        output_dir: str,
        min_duration: float = 1.0,
    ) -> List[str]:
        """从长音频中提取有效片段"""
        audio, sr = self.load_audio(file_path)
        segments = self.segment_audio(audio, sr)
        saved_paths = []

        for i, seg in enumerate(segments):
            # 跳过静音段
            rms = np.sqrt(np.mean(seg.audio**2))
            if rms < 0.01:
                continue

            out_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(file_path))[0]}_seg{i:04d}.wav"
            )
            self.save_audio(seg.audio, out_path, sr)
            saved_paths.append(out_path)

        return saved_paths
