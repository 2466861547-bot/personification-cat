"""
音频合成模块: 将文本/情绪描述合成为宠物声音
使用 Bark 模型生成宠物叫声，无 GPU/网络时降级为模拟音频
"""

import os
import socket
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

# 人类声音预设 (用于声音克隆/指定人声)
# key: 目标声线名（如 "林志玲", "檀健次", "温柔女声", "磁性男声"）
# value: Bark voice_preset (Bark 内置的 speaker 或 emotion preset)
#   - v2/en_speaker_X: Bark 内置的英文说话人
#   - v2/zh_emotion_X: Bark 中文情绪预设 (需测试可用性)
#   - 也可以指向自定义 .npz/.json 声纹 embedding 文件 (由 Bark save_pretrained 生成)
HUMAN_VOICE_PRESETS = {
    # 内置 Bark speaker (Bark v2 常用预设; 具体编号需根据实际测试/枚举)
    "林志玲": "v2/en_speaker_6",      # 温柔女声 (可调整实际编号)
    "檀健次": "v2/en_speaker_9",      # 磁性男声 (可调整实际编号)
    "温柔女声": "v2/en_speaker_2",
    "甜美女声": "v2/en_speaker_4",
    "成熟女声": "v2/en_speaker_10",
    "磁性男声": "v2/en_speaker_1",
    "阳光男声": "v2/en_speaker_7",
    # 默认 Bark speaker
    "default": "v2/en_speaker_0",
}

# 中文声音描述 (用于 Bark 中文生成时的 prompt)
CHINESE_VOICE_DESCRIPTION = "一个人在说话"

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
    "curious": "宠物发出好奇的咕咕声，声音轻柔而短促",
    "lonely": "宠物发出孤独的长鸣声，声音低沉而间隔较长",
    "anxious": "宠物发出焦虑的呜咽声，声音断续而不安",
    "excited": "宠物发出兴奋的快速叫声，声音连续而高亢",
    "frustrated": "宠物发出挫败的喷气声，声音短促而低沉",
    "relaxed": "宠物发出放松的轻柔叹气声，声音平稳而舒缓",
    "territorial": "宠物发出宣示领地的嚎叫声，声音中低频而持续",
    "greeting": "宠物发出问候的欢快叫声，声音轻快而短促",
    "confused": "宠物发出困惑的犹豫叫声，声音忽高忽低而间歇",
    "jealous": "宠物发出嫉妒的插嘴叫声，声音急促而抢断",
}


class AudioGenerator:
    """宠物声音合成器 - 支持优雅降级"""

    def __init__(self, model_name: str = "suno/bark"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.is_ready = False
        self._generation_config_fixed = False  # 只修一次
        self._load_model()

    def _load_model(self):
        """加载 Bark 模型 - 三级降级: 本地缓存 → HF → ModelScope → 模拟音频"""
        try:
            from transformers import BarkModel, BarkProcessor

            print(f"加载 Bark 模型: {self.model_name}")

            # ===== 检查所有可能的本地缓存位置 =====
            hf_cache_dir = os.path.expanduser(
                f"~/.cache/huggingface/hub/models--{self.model_name.replace('/', '--')}"
            )
            ms_cache_dir = os.path.expanduser(
                "~/.cache/modelscope/models/mapjack--bark"
            )
            
            # ModelScope 缓存 (优先级最高，因为国内下载最快)
            if os.path.isdir(ms_cache_dir):
                snapshots_dir = os.path.join(ms_cache_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    snapshots = os.listdir(snapshots_dir)
                    if snapshots:
                        ms_model_path = os.path.join(snapshots_dir, snapshots[0])
                        print(f"  ✅ 检测到 ModelScope 本地缓存: {ms_model_path}")
                        self.processor = BarkProcessor.from_pretrained(ms_model_path)
                        self.model = BarkModel.from_pretrained(
                            ms_model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        )
                        if torch.cuda.is_available():
                            self.model = self.model.to("cuda")
                        self.is_ready = True
                        print(f"  ✅ Bark 模型加载成功 (来源: ModelScope 本地缓存)")
                        return

            # HuggingFace 缓存
            if os.path.isdir(hf_cache_dir):
                print(f"  ✅ 检测到 HuggingFace 本地缓存, 直接加载")
                self.processor = BarkProcessor.from_pretrained(self.model_name)
                self.model = BarkModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                self.is_ready = True
                print(f"  ✅ Bark 模型加载成功 (来源: HuggingFace 本地缓存)")
                return

            # 没有任何缓存，需要下载
            print(f"  ⚠️  本地无缓存, 尝试在线下载...")

            # ===== 第二级: HF 镜像 (hf-mirror.com) =====
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            try:
                self.processor = BarkProcessor.from_pretrained(self.model_name)
                self.model = BarkModel.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                self.is_ready = True
                print(f"  ✅ Bark 模型加载成功 (来源: HuggingFace 镜像)")
                return
            except Exception as hf_err:
                print(f"  ⚠️  HF 镜像下载失败: {str(hf_err)[:80]}...")

            # ===== 第三级: ModelScope 国内镜像 =====
            print(f"  🔄 尝试 ModelScope 国内镜像 (mapjack/bark)...")
            try:
                from modelscope import snapshot_download
                ms_model_path = snapshot_download("mapjack/bark")
                print(f"  ✅ ModelScope 下载成功: {ms_model_path}")
                self.processor = BarkProcessor.from_pretrained(ms_model_path)
                self.model = BarkModel.from_pretrained(
                    ms_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                if torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                self.is_ready = True
                print(f"  ✅ Bark 模型加载成功 (来源: ModelScope 国内镜像)")
                return
            except ImportError:
                print(f"  ⚠️  modelscope 未安装, 尝试安装...")
                import subprocess
                subprocess.run(["pip", "install", "modelscope", "-q"], check=False)
                try:
                    from modelscope import snapshot_download as ms_download
                    ms_model_path = ms_download("mapjack/bark")
                    print(f"  ✅ ModelScope 下载成功: {ms_model_path}")
                    self.processor = BarkProcessor.from_pretrained(ms_model_path)
                    self.model = BarkModel.from_pretrained(
                        ms_model_path,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                    if torch.cuda.is_available():
                        self.model = self.model.to("cuda")
                    self.is_ready = True
                    print(f"  ✅ Bark 模型加载成功 (来源: ModelScope 国内镜像)")
                    return
                except Exception as ms_err2:
                    print(f"  ⚠️  ModelScope 安装后仍失败: {str(ms_err2)[:80]}...")
            except Exception as ms_err:
                print(f"  ⚠️  ModelScope 下载失败: {str(ms_err)[:80]}...")

            # ===== 第四级: 降级为模拟音频 =====
            raise RuntimeError("所有下载源均不可用")

        except ImportError:
            print("⚠️  transformers 库版本不支持 Bark，降级为模拟音频生成")
            print("   如需真正的 Bark 语音合成: pip install transformers>=4.36")
            self.model = None
            self.is_ready = False
        except Exception as e:
            print(f"⚠️  Bark 模型加载失败: {e}")
            print("   降级为模拟音频生成 (使用合成波形模拟宠物叫声)")
            print("   如需真正的 Bark 语音合成:")
            print(f"     1. 连接网络后重试")
            print(f"     2. ModelScope 国内镜像: pip install modelscope && python -c \"from modelscope import snapshot_download; snapshot_download('mapjack/bark')\"")
            print(f"     3. HF 镜像: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download {self.model_name}")
            print(f"     4. 下载后存放到 ~/.cache/huggingface/hub/ 目录")
            self.model = None
            self.processor = None
            self.is_ready = False

    def get_preset(self, pet_type: str, emotion: str) -> str:
        """获取声音预设"""
        presets = VOICE_PRESETS.get(pet_type, {})
        return presets.get(emotion, presets.get("seek_attention", "v2/en_speaker_0"))

    def get_human_preset(self, target_voice: str) -> str:
        """获取人类声音预设 (支持自定义声纹克隆文件路径)"""
        # 如果目标声线是文件路径 (自定义声纹 embedding)，直接返回
        if target_voice and (target_voice.endswith(".json") or target_voice.endswith(".npz")):
            return target_voice
        # 从预设字典查找
        if target_voice in HUMAN_VOICE_PRESETS:
            return HUMAN_VOICE_PRESETS[target_voice]
        # 模糊匹配 (忽略大小写)
        for key, val in HUMAN_VOICE_PRESETS.items():
            if target_voice.lower() == key.lower():
                return val
        # 默认 fallback
        return HUMAN_VOICE_PRESETS["default"]

    def list_human_voices(self) -> list:
        """列出所有可用的人类声线"""
        voices = [k for k in HUMAN_VOICE_PRESETS.keys() if k != "default"]
        return voices

    def text_to_human_sound(
        self,
        text: str,
        target_voice: str = "default",
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        文本 → 人类声音 (支持指定声线/声纹克隆)
        Args:
            text: 要合成的文本
            target_voice: 目标声线名 (如 "林志玲", "檀健次") 或自定义 embedding 文件路径 (.json/.npz)
            output_path: 输出文件路径
        Returns:
            audio: 音频数组
        """
        if self.model is not None and self.is_ready:
            voice_preset = self.get_human_preset(target_voice)
            try:
                inputs = self.processor(
                    text,
                    voice_preset=voice_preset,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # 只在首次生成前修复 generation_config，避免重复警告
                if not self._generation_config_fixed and hasattr(self.model, "generation_config"):
                    cfg = self.model.generation_config
                    cfg.max_length = None
                    cfg.max_new_tokens = 768
                    cfg.do_sample = False
                    self._generation_config_fixed = True

                # 根据文本长度自适应调整生成 token 数
                input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 10
                max_new_tokens = min(768, max(256, input_len * 8))

                with torch.no_grad():
                    audio_array = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )
                    audio = audio_array.cpu().numpy().squeeze()
                    sample_rate = self.model.generation_config.sample_rate
                print(f"  🗣️  人类声音合成完成 (声线: {target_voice} / preset: {voice_preset})")
            except Exception as e:
                print(f"  ⚠️  Bark 人类声音合成失败: {e}，降级为模拟音频")
                audio = self._generate_mock_human_audio(text)
                sample_rate = 22050
        else:
            audio = self._generate_mock_human_audio(text)
            sample_rate = 22050

        # 保存
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            wavfile.write(output_path, sample_rate, audio.astype(np.float32))
            print(f"  🎵 音频已保存: {output_path}")

        return audio

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
        if self.model is not None and self.is_ready:
            # 使用 Bark 生成
            description = EMOTION_TO_DESCRIPTION.get(emotion, text)
            voice_preset = self.get_preset(pet_type, emotion)

            try:
                inputs = self.processor(
                    description,
                    voice_preset=voice_preset,
                )
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # 只在首次生成前修复 generation_config
                if not self._generation_config_fixed and hasattr(self.model, "generation_config"):
                    cfg = self.model.generation_config
                    cfg.max_length = None
                    cfg.max_new_tokens = 768
                    cfg.do_sample = False
                    self._generation_config_fixed = True

                # 自适应 max_new_tokens
                input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 10
                max_new_tokens = min(768, max(256, input_len * 8))

                with torch.no_grad():
                    audio_array = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                    )
                    audio = audio_array.cpu().numpy().squeeze()
                    sample_rate = self.model.generation_config.sample_rate
            except Exception as e:
                print(f"  ⚠️  Bark 生成失败: {e}，降级为模拟音频")
                audio = self._generate_mock_audio(pet_type, emotion)
                sample_rate = 22050
        else:
            # 降级为模拟音频生成 (使用合成波形)
            audio = self._generate_mock_audio(pet_type, emotion)
            sample_rate = 22050

        # 保存
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            wavfile.write(output_path, sample_rate, audio.astype(np.float32))
            print(f"  🎵 音频已保存: {output_path}")

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

    def _generate_mock_human_audio(self, text: str) -> np.ndarray:
        """生成模拟人类语音(降级 fallback) — 基于文本节奏的音调合成"""
        sr = 22050
        duration = max(0.3, len(text) * 0.08)  # 每个字约 0.08 秒
        t = np.linspace(0, duration, int(sr * duration))

        # 基频: 根据文本长度和字符能量生成自然的人声轮廓
        base_freq = 220.0  # 基础频率 (A3)
        # 根据字符数生成频率变化 (模拟语调)
        char_count = max(1, len(text))
        freq_modulation = 20.0 * np.sin(2 * np.pi * 2.0 * t)  # 语调起伏

        # 主音调: 混合基频和语调
        audio = 0.3 * np.sin(2 * np.pi * (base_freq + freq_modulation) * t)

        # 添加谐波 (让声音更像人声)
        audio += 0.1 * np.sin(2 * np.pi * (base_freq * 2 + freq_modulation * 2) * t)
        audio += 0.05 * np.sin(2 * np.pi * (base_freq * 3) * t)

        # 清音 vs 浊音: 根据字符节奏调整振幅 (模拟说话节奏)
        char_duration = duration / char_count
        amplitude = np.ones_like(t)
        for i in range(char_count):
            start = i * char_duration
            end = start + char_duration * 0.7  # 字间停顿
            mask = (t >= start) & (t < end)
            # 每个字的振幅包络 (attack-decay)
            char_env = np.exp(-((t[mask] - start) / (char_duration * 0.3)) ** 2)
            amplitude[mask] = char_env

        audio = audio * amplitude

        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.5
        return audio.astype(np.float32)
