"""
音频合成模块: 将文本/情绪描述合成为宠物声音
使用 Bark 模型生成宠物叫声
"""

import os
import torch
import numpy as np
from typing import Optional, Dict
from scipy.io import wavfile


# 宠物声音预设
VOICE_PRESETS = {
    "cat": {
        "hungry": "v2/cat_sounds_hungry",
        "happy": "v2/cat_sounds_happy",
        "angry": "v2/cat_sounds_angry",
        "fear": "v2/cat_sounds_fear",
        "seek_attention": "v2/cat_sounds_attention",
        "content": "v2/cat_sounds_content",
    },
    "dog": {
        "hungry": "v2/dog_sounds_hungry",
        "happy": "v2/dog_sounds_happy",
        "angry": "v2/dog_sounds_angry",
        "fear": "v2/dog_sounds_fear",
        "seek_attention": "v2/dog_sounds_attention",
        "playful": "v2/dog_sounds_playful",
        "alert": "v2/dog_sounds_alert",
    },
}

# 情绪到声音描述的映射
EMOTION_TO_DESCRIPTION = {
    "hungry": "宠物发出饥饿的叫声，声音急促而重复",
    "happy": "宠物发出开心的叫声，声音轻快而柔和",
    "angry": "宠物发出愤怒的叫声，声音低沉而激烈",
    "fear": "宠物发出恐惧的叫声，声音颤抖而微弱",
    "sad": "宠物发出悲伤的叫声，声音低沉而缓慢",
    "seek_attention": "宠物发出求关注的叫声，声音持续而有节奏",
    "pain": "宠物发出疼痛的叫声，声音尖锐而短促",
    "alert": "宠物发出警觉的叫声，声音急促而高亢",
    "content": "宠物发出满足的呼噜声，声音低沉而持续",
    "playful": "宠物发出想玩耍的叫声，声音轻快而活泼",
}


class AudioGenerator:
    """宠物声音合成器"""

    def __init__(self, model_name: str = "suno/bark"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 Bark 模型"""
        try:
            from transformers import BarkModel, BarkProcessor
            print(f"加载 Bark 模型: {self.model_name}")
            self.processor = BarkProcessor.from_pretrained(self.model_name)
            self.model = BarkModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except ImportError:
            print("警告: 未安装 transformers 或 Bark 模型不可用")
            self.model = None

    def get_preset(self, pet_type: str, emotion: str) -> str:
        """获取声音预设"""
        presets = VOICE_PRESETS.get(pet_type, {})
        return presets.get(emotion, presets.get("seek_attention", "v2/en_speaker_0"))

    def text_to_pet_sound(
        self,
        text: str,
        pet_type: str = "cat",
        emotion: str = "content",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        文本 → 宠物声音
        Args:
            text: 描述文本
            pet_type: cat / dog
            emotion: 情绪标签
            output_path: 输出文件路径
        Returns:
            audio: 音频数组
        """
        if self.model is None:
            # 生成模拟音频
            audio = self._generate_mock_audio(pet_type, emotion)
        else:
            # 使用 Bark 生成
            description = EMOTION_TO_DESCRIPTION.get(emotion, text)
            voice_preset = self.get_preset(pet_type, emotion)

            inputs = self.processor(
                description,
                voice_preset=voice_preset,
            )
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                audio_array = self.model.generate(**inputs)
                audio = audio_array.cpu().numpy().squeeze()
                # Bark 采样率
                sample_rate = self.model.generation_config.sample_rate

        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if self.model is None:
                wavfile.write(output_path, 22050, audio.astype(np.float32))
            else:
                wavfile.write(output_path, 24000, audio.astype(np.float32))

        return audio

    def emotion_to_pet_sound(
        self,
        emotion: str,
        pet_type: str = "cat",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """情绪 → 宠物声音"""
        description = EMOTION_TO_DESCRIPTION.get(
            emotion, f"{pet_type}发出{emotion}的声音"
        )
        return self.text_to_pet_sound(
            text=description,
            pet_type=pet_type,
            emotion=emotion,
            output_path=output_path,
        )

    def human_speech_to_pet_sound(
        self,
        speech_text: str,
        target_pet: str = "cat",
        target_emotion: str = "content",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        人类语音文本 → 宠物声音
        将人类说的话翻译成对应情绪的宠物叫声
        """
        # 1. 根据人类语句判断意图
        intent = self._analyze_intent(speech_text)

        # 2. 映射为宠物情绪
        pet_emotion = self._map_intent_to_emotion(intent, target_pet)

        # 3. 生成宠物声音
        return self.emotion_to_pet_sound(
            emotion=pet_emotion,
            pet_type=target_pet,
            output_path=output_path,
        )

    def _analyze_intent(self, text: str) -> str:
        """简单意图分析(实际应接入微调后的 LLM)"""
        text_lower = text.lower()
        if any(w in text for w in ["吃饭", "饿了", "食物", "饭", "eat", "hungry", "food"]):
            return "feeding"
        elif any(w in text for w in ["出去玩", "散步", "玩", "walk", "play", "outside"]):
            return "play"
        elif any(w in text for w in ["乖", "好孩子", "good", "good boy", "good girl"]):
            return "praise"
        elif any(w in text for w in ["不行", "不可以", "no", "stop", "bad"]):
            return "scold"
        elif any(w in text for w in ["过来", "过来", "come", "here"]):
            return "call"
        else:
            return "talk"

    def _map_intent_to_emotion(self, intent: str, pet_type: str) -> str:
        """意图 → 宠物情绪映射"""
        mapping = {
            "feeding": "hungry",
            "play": "playful",
            "praise": "happy",
            "scold": "sad",
            "call": "seek_attention",
            "talk": "content",
        }
        return mapping.get(intent, "content")

    def _generate_mock_audio(self, pet_type: str, emotion: str) -> np.ndarray:
        """生成模拟音频(无 GPU 时的 fallback)"""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        if pet_type == "cat":
            if emotion == "content":
                # 呼噜声: 低频周期
                audio = 0.3 * np.sin(2 * np.pi * 25 * t) * np.exp(-t * 0.5)
            elif emotion == "hungry":
                # 饥饿叫声: 高频短促
                audio = 0.5 * np.sin(2 * np.pi * 800 * t) * (
                    np.mod(t * 3, 1) < 0.4
                ).astype(float)
            elif emotion == "angry":
                # 嘶嘶声: 噪声
                audio = 0.4 * np.random.randn(len(t)) * (
                    np.mod(t * 2, 1) < 0.3
                ).astype(float)
            else:
                # 普通喵叫
                freq = 600 + 200 * np.sin(2 * np.pi * 5 * t)
                audio = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.3)
        else:
            # 狗
            if emotion == "happy":
                # 汪汪叫(开心)
                audio = np.zeros_like(t)
                for bark_time in [0.1, 0.4, 0.7]:
                    mask = (t > bark_time) & (t < bark_time + 0.15)
                    audio[mask] = 0.5 * np.sin(2 * np.pi * 500 * t[mask])
            elif emotion == "angry":
                # 低吼
                audio = 0.4 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))
            elif emotion == "playful":
                # 短促叫声
                audio = np.zeros_like(t)
                for bark_time in [0.1, 0.3, 0.5, 0.7]:
                    mask = (t > bark_time) & (t < bark_time + 0.1)
                    audio[mask] = 0.4 * np.sin(2 * np.pi * 700 * t[mask])
            else:
                # 普通汪叫
                audio = 0.4 * np.sin(2 * np.pi * 400 * t) * np.exp(-t * 0.5)

        return audio.astype(np.float32)
