"""
逗猫逗狗娱乐模块: 相声 / 唱歌 / 安抚音
- 相声模式: 给宠物说相声段子
- 唱歌模式: 给宠物唱歌逗乐
- 安抚音模式: 播放让宠物放松的声音
- 互动模式: 根据宠物反应动态调整
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from scipy.io import wavfile


class EntertainmentMode(Enum):
    """娱乐模式"""
    XIANGSHENG = "xiangsheng"     # 相声
    SINGING = "singing"           # 唱歌
    SOOTHING = "soothing"        # 安抚音
    TEASING = "teasing"          # 逗乐
    STORYTELLING = "storytelling" # 讲故事


# =====================================
# 相声段子库
# =====================================

XIANGSHENG_SCRIPTS = {
    "cat": [
        {
            "title": "猫咪的烦恼",
            "content": [
                "哎，我跟你说啊，我家那只猫，最近可有意思了",
                "怎么了？",
                "它啊，每天准时五点半叫我起床，比闹钟还准",
                "那不是挺好嘛，省得买闹钟了",
                "好什么好！周末也想睡懒觉啊！它不管，五点半必叫",
                "哈哈，那它怎么叫？",
                "先是用爪子拍我脸，拍不醒就坐我胸口上喵喵叫",
                "这招厉害",
                "最过分的是，有次它把一杯水给我打翻在脸上",
                "哈哈哈哈，这猫成精了",
            ],
        },
        {
            "title": "猫咪选主人",
            "content": [
                "你知道猫咪是怎么选主人的吗？",
                "不知道，怎么选？",
                "它往地上一躺，看谁先过来摸它，谁就是主人",
                "这么简单？",
                "简单？那你想想，十个人围着它，就你能摸",
                "那是，猫选人跟选妃似的",
                "而且啊，选完了还后悔，第二天可能换一个",
                "哈哈，渣猫！",
            ],
        },
        {
            "title": "猫和纸箱",
            "content": [
                "我家猫啊，我给它买了个八百块的猫窝，它不睡",
                "那它睡哪？",
                "睡装猫窝的纸箱！八百块的窝它看都不看一眼",
                "经典，猫和纸箱是真爱",
                "我后来学聪明了，买了个快递，纸箱留着",
                "它睡了吗？",
                "睡了！而且啊，同一个纸箱它睡腻了还要换新的",
                "所以你的意思是，猫比你还费钱？",
                "可不是嘛！我的快递箱全归它了",
            ],
        },
    ],
    "dog": [
        {
            "title": "狗狗的忠诚",
            "content": [
                "你知道狗为什么是人类最好的朋友吗？",
                "为什么？",
                "因为你出门五分钟回来，它高兴得像你出去了五年",
                "哈哈，真的",
                "猫呢？你出门五年回来，它看你一眼，意思是",
                "意思是什么？",
                "哟，你还活着呢？碗里的猫粮没了，去添",
                "哈哈哈，太真实了",
            ],
        },
        {
            "title": "遛狗的学问",
            "content": [
                "遛狗这事儿啊，学问大了",
                "怎么说？",
                "你以为是你遛狗？其实是狗遛你",
                "此话怎讲？",
                "它往东你不敢往西，它停你就得停，它跑你就得追",
                "那到底谁遛谁？",
                "看谁牵绳就知道了，它牵着你走呢",
                "哈哈，低头看看，果然是它带我走",
            ],
        },
    ],
}


# =====================================
# 唱歌曲库
# =====================================

SINGING_SONGS = {
    "cat": [
        {
            "title": "小猫咪之歌",
            "lyrics": "小猫咪~ 小猫咪~ 你为什么这么可爱~\n"
                     "毛茸茸~ 软乎乎~ 让人想抱一抱~\n"
                     "喵喵喵~ 喵喵喵~ 快过来让我摸摸~\n"
                     "小猫咪~ 小猫咪~ 你是我的小宝贝~\n",
            "melody_freq": [523, 587, 659, 587, 523, 440, 392],  # C大调音阶
            "tempo": 80,
        },
        {
            "title": "呼噜呼噜摇篮曲",
            "lyrics": "呼噜呼噜~ 小猫咪~ 睡吧睡吧~ \n"
                     "暖暖的太阳~ 照着你~ \n"
                     "呼噜呼噜~ 别害怕~ 我在身边~ \n"
                     "做个好梦~ 小宝贝~ \n",
            "melody_freq": [392, 440, 494, 440, 392, 349, 330],
            "tempo": 60,
        },
        {
            "title": "逗猫进行曲",
            "lyrics": "来来来~ 小猫咪~ 看这里看这里~\n"
                     "逗猫棒~ 晃一晃~ 来抓呀来抓呀~\n"
                     "跳一跳~ 扑一扑~ 好厉害好厉害~\n"
                     "小猫咪~ 真棒呀~ 再来一次好不好~\n",
            "melody_freq": [659, 698, 784, 698, 659, 587, 523],
            "tempo": 120,
        },
    ],
    "dog": [
        {
            "title": "汪汪之歌",
            "lyrics": "汪汪汪~ 小狗狗~ 你是我的好朋友~\n"
                     "摇摇尾~ 转个圈~ 快乐每一天~\n"
                     "汪汪汪~ 去散步~ 草地上去跑跑~\n"
                     "小狗狗~ 小狗狗~ 永远在一起~\n",
            "melody_freq": [523, 587, 659, 587, 523, 440, 392],
            "tempo": 100,
        },
        {
            "title": "好狗狗摇篮曲",
            "lyrics": "好狗狗~ 乖狗狗~ 该睡觉啦~ \n"
                     "月光下~ 星星闪~ 晚安宝贝~ \n"
                     "明天呢~ 去公园~ 跑跑跳跳~ \n"
                     "现在呢~ 闭上眼~ 做个好梦~ \n",
            "melody_freq": [392, 440, 494, 440, 392, 349, 330],
            "tempo": 50,
        },
    ],
}


# =====================================
# 安抚音库
# =====================================

SOOTHING_SOUNDS = {
    "cat": {
        "purr_simulation": {
            "description": "模拟猫咪呼噜声,25-50Hz 低频",
            "freq": 25,
            "modulation_freq": 2,
            "duration": 300,  # 5分钟
        },
        "heartbeat": {
            "description": "心跳声,模拟母猫怀抱",
            "freq": 60,
            "modulation_freq": 1.2,
            "duration": 300,
        },
        "rain_sound": {
            "description": "雨声,白噪音安抚",
            "noise_type": "pink",
            "duration": 600,
        },
        "bird_chirp": {
            "description": "轻柔鸟鸣,吸引注意力",
            "freq_range": (2000, 4000),
            "duration": 180,
        },
    },
    "dog": {
        "heartbeat": {
            "description": "心跳声,模拟母犬陪伴",
            "freq": 60,
            "modulation_freq": 1.0,
            "duration": 300,
        },
        "rain_sound": {
            "description": "雨声白噪音,缓解焦虑",
            "noise_type": "pink",
            "duration": 600,
        },
        "classical_music": {
            "description": "古典音乐片段,研究表明狗狗偏好",
            "melody_freq": [261, 293, 329, 349, 392, 440, 493],
            "tempo": 60,
            "duration": 300,
        },
        "human_whisper": {
            "description": "人类轻语'好狗狗',安抚情绪",
            "freq_range": (200, 400),
            "duration": 180,
        },
    },
}


class EntertainmentEngine:
    """宠物娱乐引擎"""

    def __init__(self, audio_generator=None):
        self.audio_generator = audio_generator
        self.sr = 22050

    # ==============================
    # 相声模式
    # ==============================

    def play_xiangsheng(
        self,
        pet_type: str = "cat",
        script_index: int = 0,
        output_path: str = "",
    ) -> Dict:
        """说相声给宠物听"""
        scripts = XIANGSHENG_SCRIPTS.get(pet_type, XIANGSHENG_SCRIPTS["cat"])

        if script_index >= len(scripts):
            script_index = 0

        script = scripts[script_index]

        # 生成相声音频 (模拟双人对话)
        audio = self._generate_xiangsheng_audio(script)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, self.sr, audio)

        return {
            "mode": "xiangsheng",
            "pet_type": pet_type,
            "title": script["title"],
            "script": script["content"],
            "output_path": output_path,
            "duration": len(audio) / self.sr,
        }

    def _generate_xiangsheng_audio(self, script: Dict) -> np.ndarray:
        """生成相声音频(模拟说话声调)"""
        audio_segments = []

        for i, line in enumerate(script["content"]):
            # 交替声调 (逗哏 vs 捧哏)
            if i % 2 == 0:
                # 逗哏: 较高声调
                base_freq = 200 + np.random.randint(-20, 20)
            else:
                # 捧哏: 较低声调
                base_freq = 150 + np.random.randint(-15, 15)

            # 根据文本长度决定时长
            duration = max(1.0, len(line) * 0.15)
            t = np.linspace(0, duration, int(self.sr * duration))

            # 生成语调(模拟人声基频变化)
            freq_variation = base_freq + 30 * np.sin(2 * np.pi * 3 * t)
            wave = 0.3 * np.sin(2 * np.pi * freq_variation * t)

            # 加入谐波 (模拟人声)
            wave += 0.1 * np.sin(2 * np.pi * 2 * freq_variation * t)
            wave += 0.05 * np.sin(2 * np.pi * 3 * freq_variation * t)

            # 包络 (说话节奏)
            envelope = np.exp(-t * 0.5) * (1 - np.exp(-t * 10))
            wave = wave * envelope

            # 停顿
            silence = np.zeros(int(self.sr * 0.3))

            audio_segments.append(wave)
            audio_segments.append(silence)

        audio = np.concatenate(audio_segments)
        return audio.astype(np.float32)

    # ==============================
    # 唱歌模式
    # ==============================

    def play_singing(
        self,
        pet_type: str = "cat",
        song_index: int = 0,
        output_path: str = "",
    ) -> Dict:
        """唱歌给宠物听"""
        songs = SINGING_SONGS.get(pet_type, SINGING_SONGS["cat"])

        if song_index >= len(songs):
            song_index = 0

        song = songs[song_index]
        audio = self._generate_singing_audio(song)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, self.sr, audio)

        return {
            "mode": "singing",
            "pet_type": pet_type,
            "title": song["title"],
            "lyrics": song["lyrics"],
            "output_path": output_path,
            "duration": len(audio) / self.sr,
        }

    def _generate_singing_audio(self, song: Dict) -> np.ndarray:
        """生成唱歌音频"""
        melody = song["melody_freq"]
        tempo = song.get("tempo", 80)
        beat_duration = 60.0 / tempo  # 每拍时长

        audio_segments = []
        note_index = 0
        lyrics_lines = song["lyrics"].split("\n")

        for line in lyrics_lines:
            for char in line:
                if char in ["~", " ", "\n"]:
                    # 延音
                    if audio_segments:
                        last = audio_segments[-1]
                        ext = np.zeros(int(self.sr * beat_duration * 0.3))
                        audio_segments.append(ext)
                    continue

                # 取音符频率
                freq = melody[note_index % len(melody)]
                note_index += 1

                # 生成一个音符
                t = np.linspace(0, beat_duration * 0.8, int(self.sr * beat_duration * 0.8))

                # 基频 + 谐波 (模拟人声)
                wave = 0.3 * np.sin(2 * np.pi * freq * t)
                wave += 0.15 * np.sin(2 * np.pi * 2 * freq * t)
                wave += 0.05 * np.sin(2 * np.pi * 3 * freq * t)

                # 颤音 (vibrato)
                vibrato = 5 * np.sin(2 * np.pi * 5 * t)
                wave = 0.3 * np.sin(2 * np.pi * (freq + vibrato) * t)

                # ADSR 包络
                attack = int(0.1 * len(t))
                decay = int(0.2 * len(t))
                release = int(0.2 * len(t))

                envelope = np.ones_like(t)
                envelope[:attack] = np.linspace(0, 1, attack)
                envelope[-release:] = np.linspace(1, 0, release)

                wave = wave * envelope
                audio_segments.append(wave)

            # 换行停顿
            audio_segments.append(np.zeros(int(self.sr * 0.3)))

        audio = np.concatenate(audio_segments) if audio_segments else np.zeros(self.sr)
        return audio.astype(np.float32)

    # ==============================
    # 安抚音模式
    # ==============================

    def play_soothing(
        self,
        pet_type: str = "cat",
        sound_type: str = "purr_simulation",
        duration: int = 300,
        output_path: str = "",
    ) -> Dict:
        """播放安抚音"""
        sounds = SOOTHING_SOUNDS.get(pet_type, SOOTHING_SOUNDS["cat"])

        if sound_type not in sounds:
            sound_type = list(sounds.keys())[0]

        sound_config = sounds[sound_type]
        audio = self._generate_soothing_audio(sound_config, duration)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, self.sr, audio)

        return {
            "mode": "soothing",
            "pet_type": pet_type,
            "sound_type": sound_type,
            "description": sound_config["description"],
            "output_path": output_path,
            "duration": len(audio) / self.sr,
        }

    def _generate_soothing_audio(
        self, config: Dict, duration: int
    ) -> np.ndarray:
        """生成安抚音"""
        sr = self.sr
        total_samples = int(sr * duration)
        t = np.linspace(0, duration, total_samples)

        sound_type = config.get("description", "")

        if "呼噜" in sound_type or "purr" in sound_type.lower():
            # 呼噜声: 25Hz 低频 + 调制
            base_freq = config.get("freq", 25)
            mod_freq = config.get("modulation_freq", 2)
            audio = 0.3 * np.sin(2 * np.pi * base_freq * t) * \
                    (0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t))

        elif "心跳" in sound_type or "heartbeat" in sound_type.lower():
            # 心跳声: 周期性低频脉冲
            freq = config.get("freq", 60)
            mod_freq = config.get("modulation_freq", 1.2)
            period = 1.0 / mod_freq

            audio = np.zeros(total_samples)
            for beat_time in np.arange(0, duration, period):
                beat_start = int(beat_time * sr)
                beat_len = int(0.15 * sr)  # 每次心跳持续150ms
                if beat_start + beat_len < total_samples:
                    beat_t = np.linspace(0, 0.15, beat_len)
                    beat = 0.4 * np.sin(2 * np.pi * freq * beat_t) * \
                           np.exp(-beat_t * 15)
                    audio[beat_start:beat_start + beat_len] = beat

        elif "雨" in sound_type or "rain" in sound_type.lower():
            # 雨声: 粉噪声 + 低通滤波
            white = np.random.randn(total_samples)
            # 简易低通滤波
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 2000 / (sr / 2), btype="low")
            audio = 0.2 * filtfilt(b, a, white)

        elif "鸟" in sound_type or "bird" in sound_type.lower():
            # 鸟鸣: 高频啁啾
            freq_range = config.get("freq_range", (2000, 4000))
            audio = np.zeros(total_samples)

            chirp_interval = 2.0  # 每2秒一次
            for chirp_time in np.arange(0, duration, chirp_interval):
                start = int(chirp_time * sr)
                chirp_len = int(0.3 * sr)  # 0.3秒
                if start + chirp_len < total_samples:
                    chirp_t = np.linspace(0, 0.3, chirp_len)
                    freq = np.random.uniform(freq_range[0], freq_range[1])
                    freq_sweep = freq + 500 * np.sin(2 * np.pi * 10 * chirp_t)
                    chirp = 0.2 * np.sin(2 * np.pi * freq_sweep * chirp_t) * \
                            np.exp(-chirp_t * 3)
                    audio[start:start + chirp_len] = chirp

        elif "古典" in sound_type or "classical" in sound_type.lower():
            # 古典音乐片段
            melody = config.get("melody_freq", [261, 293, 329, 349, 392])
            tempo = config.get("tempo", 60)
            beat_duration = 60.0 / tempo

            audio = np.zeros(total_samples)
            pos = 0
            while pos < total_samples:
                for freq in melody:
                    note_len = int(beat_duration * sr)
                    if pos + note_len < total_samples:
                        note_t = np.linspace(0, beat_duration, note_len)
                        wave = 0.2 * np.sin(2 * np.pi * freq * note_t) * \
                               np.exp(-note_t * 1)
                        audio[pos:pos + note_len] = wave
                    pos += note_len

        elif "轻语" in sound_type or "whisper" in sound_type.lower():
            # 人声轻语
            freq_range = config.get("freq_range", (200, 400))
            audio = np.zeros(total_samples)

            phrase_interval = 5.0
            for phrase_time in np.arange(0, duration, phrase_interval):
                start = int(phrase_time * sr)
                phrase_len = int(2.0 * sr)
                if start + phrase_len < total_samples:
                    phrase_t = np.linspace(0, 2.0, phrase_len)
                    freq = np.random.uniform(freq_range[0], freq_range[1])
                    freq_var = freq + 50 * np.sin(2 * np.pi * 3 * phrase_t)
                    wave = 0.15 * np.sin(2 * np.pi * freq_var * phrase_t) * \
                           (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * phrase_t))
                    audio[start:start + phrase_len] = wave

        else:
            # 默认: 粉噪声
            audio = 0.1 * np.random.randn(total_samples)

        return audio.astype(np.float32)

    # ==============================
    # 逗乐模式
    # ==============================

    def play_teasing(
        self,
        pet_type: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """逗乐音 (模仿猎物声音引起宠物兴趣)"""
        sr = self.sr
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))

        if pet_type == "cat":
            # 模仿小老鼠/虫子的声音
            audio = np.zeros_like(t)
            # 高频短促声 (模拟啮齿动物)
            for start_time in np.arange(0.5, duration, 1.5 + np.random.random()):
                start = int(start_time * sr)
                chirp_len = int(0.1 * sr)
                if start + chirp_len < len(audio):
                    chirp_t = np.linspace(0, 0.1, chirp_len)
                    freq = np.random.uniform(3000, 5000)
                    chirp = 0.2 * np.sin(2 * np.pi * freq * chirp_t) * \
                            np.exp(-chirp_t * 20)
                    audio[start:start + chirp_len] = chirp

            # 加入沙沙声 (模拟纸袋)
            rustle = 0.05 * np.random.randn(len(t))
            from scipy.signal import butter, filtfilt
            b, a = butter(4, 5000 / (sr / 2), btype="high")
            rustle = filtfilt(b, a, rustle)
            audio += rustle

        else:
            # 狗: 模仿球弹跳声 + 口哨
            audio = np.zeros_like(t)
            for start_time in np.arange(0.5, duration, 1.0):
                start = int(start_time * sr)
                bounce_len = int(0.2 * sr)
                if start + bounce_len < len(audio):
                    bounce_t = np.linspace(0, 0.2, bounce_len)
                    freq = 400 + 200 * np.exp(-bounce_t * 10)
                    bounce = 0.3 * np.sin(2 * np.pi * freq * bounce_t) * \
                             np.exp(-bounce_t * 8)
                    audio[start:start + bounce_len] = bounce

            # 口哨声
            whistle_t = t.copy()
            whistle_freq = 2000 + 500 * np.sin(2 * np.pi * 0.5 * whistle_t)
            whistle = 0.1 * np.sin(2 * np.pi * whistle_freq * whistle_t) * \
                      (np.mod(whistle_t, 3) < 1).astype(float)
            audio += whistle

        audio = audio / (np.max(np.abs(audio)) + 1e-10) * 0.8

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, sr, audio)

        return {
            "mode": "teasing",
            "pet_type": pet_type,
            "output_path": output_path,
            "duration": duration,
        }

    # ==============================
    # 讲故事模式
    # ==============================

    def play_storytelling(
        self,
        pet_type: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """讲故事给宠物听"""
        stories = {
            "cat": "从前有一只小猫咪，住在一个温暖的大房子里。"
                   "每天早上，太阳公公升起来的时候，小猫咪就会伸一个大大的懒腰。"
                   "然后呢，它会走到主人床边，轻轻地喵喵叫，叫主人起床。"
                   "主人就会摸摸它的小脑袋，说，早安呀小猫咪。"
                   "小猫咪开心地呼噜呼噜，开始新的一天。"
                   "它跑到窗台上，看窗外的小鸟叽叽喳喳。"
                   "然后跑到厨房，看主人准备早餐。"
                   "吃完饭，它就找一个有太阳的地方，晒着太阳睡个午觉。"
                   "这就是小猫咪幸福的一天。",
            "dog": "从前有一只小狗狗，它有一个最好的人类朋友。"
                   "每天早上，主人一打开房门，小狗狗就摇着尾巴冲出去。"
                   "它在草地上跑啊跑，追蝴蝶，闻花香，开心极了。"
                   "主人扔出一个球，小狗狗嗖地跑过去，叼回来给主人。"
                   "主人说，好狗狗！小狗狗更开心了。"
                   "散步回来，主人给它准备了好吃早餐。"
                   "吃完饭，小狗狗趴在主人脚边，打了一个大哈欠。"
                   "主人摸着它的头说，你是世界上最好的狗狗。"
                   "小狗狗摇摇尾巴，幸福地睡着了。",
        }

        story = stories.get(pet_type, stories["cat"])

        # 生成故事音频 (模拟温柔说话声)
        audio = self._generate_storytelling_audio(story)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            wavfile.write(output_path, self.sr, audio)

        return {
            "mode": "storytelling",
            "pet_type": pet_type,
            "story": story,
            "output_path": output_path,
            "duration": len(audio) / self.sr,
        }

    def _generate_storytelling_audio(self, text: str) -> np.ndarray:
        """生成讲故事音频(温柔女声风格)"""
        sr = self.sr
        segments = []

        for char in text:
            if char in ["，", "。", "；", "！", "？"]:
                # 停顿
                segments.append(np.zeros(int(sr * 0.3)))
                continue
            elif char in [" ", "\n"]:
                segments.append(np.zeros(int(sr * 0.1)))
                continue

            # 每个字一个音节
            duration = 0.2
            t = np.linspace(0, duration, int(sr * duration))

            # 女声基频 (180-220Hz)
            base_freq = 200 + np.random.randint(-15, 15)
            # 语调变化
            freq_var = base_freq + 20 * np.sin(2 * np.pi * 2 * t)

            wave = 0.25 * np.sin(2 * np.pi * freq_var * t)
            wave += 0.1 * np.sin(2 * np.pi * 2 * freq_var * t)
            wave += 0.05 * np.sin(2 * np.pi * 3 * freq_var * t)

            # 包络
            envelope = np.exp(-t * 2) * (1 - np.exp(-t * 20))
            wave = wave * envelope

            segments.append(wave)

        audio = np.concatenate(segments) if segments else np.zeros(sr)
        return audio.astype(np.float32)

    # ==============================
    # 获取模式列表
    # ==============================

    def list_modes(self, pet_type: str = "cat") -> Dict:
        """列出所有可用模式"""
        return {
            "xiangsheng": {
                "name": "相声模式",
                "description": "给宠物说相声段子，逗宠物开心",
                "scripts": [s["title"] for s in XIANGSHENG_SCRIPTS.get(pet_type, [])],
            },
            "singing": {
                "name": "唱歌模式",
                "description": "唱歌给宠物听，旋律轻快或舒缓",
                "songs": [s["title"] for s in SINGING_SONGS.get(pet_type, [])],
            },
            "soothing": {
                "name": "安抚音模式",
                "description": "播放让宠物放松的声音(呼噜声/心跳/雨声等)",
                "sounds": list(SOOTHING_SOUNDS.get(pet_type, {}).keys()),
            },
            "teasing": {
                "name": "逗乐模式",
                "description": "模仿猎物声音吸引宠物注意力",
            },
            "storytelling": {
                "name": "讲故事模式",
                "description": "用温柔声音给宠物讲故事",
            },
        }
