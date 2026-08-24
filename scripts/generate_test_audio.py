#!/usr/bin/env python3
"""生成真实感测试音频 - 猫叫/狗叫/人类语音 (多种情绪)
使用真实声学特征模拟真实宠物叫声音频 (16kHz 单声道 WAV)
"""
import os
import numpy as np
from scipy.io import wavfile

SR = 16000

# ============ 工具函数 ============
def envelope(n_samples, attack=0.05, release=0.1, sr=SR):
    """生成ADSR包络"""
    attack_s = int(attack * sr)
    release_s = int(release * sr)
    sustain_s = n_samples - attack_s - release_s
    if sustain_s < 0:
        attack_s = int(n_samples * 0.3)
        release_s = n_samples - attack_s
        sustain_s = 0
    env = np.concatenate([
        np.linspace(0, 1, attack_s),
        np.ones(sustain_s) if sustain_s > 0 else np.array([]),
        np.linspace(1, 0, release_s),
    ])
    return env[:n_samples]

def add_noise(audio, level=0.005):
    """添加环境噪声模拟真实录音"""
    return audio + np.random.randn(len(audio)) * level

def meow(note_hz, duration, vibrato=True, sr=SR):
    """合成猫叫: 带颤音、FM调制的元音化叫声
    真实猫叫: 基频 250-800Hz, 第一共振峰 ~600Hz, 第二 ~1100Hz
    """
    t = np.arange(0, duration, 1/sr)
    # 基频随时间变化 (上升再下降)
    f0_start = note_hz * 0.95
    f0_mid = note_hz * 1.08
    f0_end = note_hz * 0.85
    f0_env = np.interp(np.linspace(0, 1, len(t)), [0, 0.4, 1.0], [f0_start, f0_mid, f0_end])
    if vibrato:
        f0_env = f0_env * (1 + 0.02 * np.sin(2 * np.pi * 6 * t))  # 6Hz颤音, 2%深度
    
    phase = np.cumsum(2 * np.pi * f0_env / sr)
    # 多个谐波 (猫叫有丰富谐波结构)
    harmonics = [1.0, 0.55, 0.35, 0.2, 0.12, 0.06]
    sig = np.zeros_like(t)
    for i, amp in enumerate(harmonics, 1):
        sig += amp * np.sin(i * phase)
    
    # 共振峰滤波 (formants): F1 ~650, F2 ~1100Hz
    # 简单低通模拟
    sig = np.convolve(sig, np.exp(-np.arange(10)/3), mode='same') / 2.0
    # 包络
    env = envelope(len(t), attack=0.08, release=0.15, sr=sr)
    sig = sig * env
    # 归一化
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig)) * 0.85
    return sig

def bark_sound(note_hz, duration, n_barks=1, pitch_drop=True, sr=SR):
    """合成狗叫: 短促爆发性叫声, 低高频叠加
    真实狗叫: 基频 200-600Hz, 叫间距 0.15-0.4s
    每个 bark 单声占总时长 ~60%，间距 40%
    """
    # 总时长分摊: 每个 bark 的单叫时长 (考虑总时长和间距)
    # total = n_barks * (bark_len + gap)  - gap (最后一个没有gap)
    # bark_len 占比 60%, gap 40% per pair
    ratio_bark = 0.6
    if n_barks == 1:
        single_dur = duration
        gap = 0
    else:
        gap = duration * (1-ratio_bark) / (n_barks - 1)
        single_dur = (duration - (n_barks-1)*gap) / n_barks
        if single_dur < 0.08:  # 最少 80ms
            single_dur = 0.08
            gap = max(0, (duration - n_barks*single_dur) / max(1, n_barks-1))
    segments = []
    for i in range(n_barks):
        t = np.arange(0, single_dur, 1/sr)
        # 狗叫基频随时间快速下降
        if pitch_drop:
            f0_env = np.linspace(note_hz * 1.15, note_hz * 0.8, len(t))
        else:
            f0_env = np.ones(len(t)) * note_hz
        phase = np.cumsum(2 * np.pi * f0_env / sr)
        # 谐波: 狗叫有大量高频泛音
        harmonics = [1.0, 0.7, 0.45, 0.25, 0.12, 0.06, 0.03]
        sig = np.zeros_like(t)
        for j, amp in enumerate(harmonics, 1):
            sig += amp * np.sin(j * phase)
        # 尖锐高频噪声 (模拟吠叫的爆发感)
        sig += 0.15 * np.random.randn(len(t)) * envelope(len(t), attack=0.005, release=0.04, sr=sr)
        # 包络: 极短attack
        env = envelope(len(t), attack=0.008, release=single_dur * 0.55, sr=sr)
        sig = sig * env
        if np.max(np.abs(sig)) > 0:
            sig = sig / np.max(np.abs(sig)) * 0.9
        segments.append(sig)
        if i < n_barks - 1:
            segments.append(np.zeros(int(gap * sr)))
    
    return np.concatenate(segments)

def human_speech_synth(sentence, duration, sr=SR):
    """模拟人类语音: 使用共振峰扫描模拟元音串
    只用于音频存在性测试, 不追求真实可懂度
    """
    t = np.arange(0, duration, 1/sr)
    n_chars = max(3, len(sentence))
    # 中文声母/韵母的平均共振峰
    # a: F1=800 F2=1200,  i: F1=300 F2=2300,  u: F1=350 F2=900
    vowels_f1 = [800, 500, 300, 350, 600, 900, 450]
    vowels_f2 = [1200, 1800, 2300, 900, 1500, 1100, 1600]
    
    base_f0 = 180  # 人声基频 ~180Hz
    n = len(t)
    f0 = base_f0 * (1 + 0.02 * np.sin(2 * np.pi * 4.5 * t))  # 4.5Hz颤音
    phase = np.cumsum(2 * np.pi * f0 / sr)
    
    sig = np.sin(phase) + 0.5 * np.sin(2*phase) + 0.25*np.sin(3*phase)
    # 用变化的共振峰模拟说话
    char_idx = (np.linspace(0, n_chars-1, n)).astype(int)
    f1_arr = np.array([vowels_f1[i % len(vowels_f1)] for i in char_idx])
    f2_arr = np.array([vowels_f2[i % len(vowels_f2)] for i in char_idx])
    
    # 带通滤波模拟 (简单二阶近似, 通过乘sin调制)
    for freq, q in [(f1_arr.mean(), 0.1), (f2_arr.mean(), 0.08)]:
        sig = sig * (1 + 0.3 * np.sin(2 * np.pi * freq * t / 3))
    
    # 音节包络
    env_syllable = 0.5 + 0.5 * np.sin(2 * np.pi * (n_chars/2/duration) * t)
    env_words = envelope(n, attack=0.03, release=0.08, sr=sr)
    sig = sig * env_syllable * env_words
    
    if np.max(np.abs(sig)) > 0:
        sig = sig / np.max(np.abs(sig)) * 0.8
    return sig

def save_wav(audio, path, sr=SR):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wavfile.write(path, sr, (add_noise(audio, level=0.004) * 32767).astype(np.int16))
    size_kb = os.path.getsize(path) / 1024
    dur = len(audio) / sr
    print(f"  ✅ {os.path.basename(path):50s} {dur:4.1f}s  {size_kb:7.1f} KB")

def analyze(audio, sr=SR):
    """分析音频并返回统计信息"""
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    main_freq = float(freqs[int(np.argmax(spec))])
    top3 = [float(f) for f in freqs[np.argsort(spec)[-3:][::-1]]]
    return peak, rms, main_freq, top3


# ============ 猫叫数据集 (12个, 覆盖 12 种情绪) ============
def gen_cat_dataset(base_dir):
    print("\n🐱 生成猫叫测试数据集 (12 个文件)...")
    print("=" * 60)
    files = []

    # hungry: 短促重复喵叫, 基频偏高
    seg1 = meow(580, 0.35)
    seg1 = np.concatenate([seg1, np.zeros(int(0.18*SR))])
    seg2 = meow(620, 0.40)
    seg2 = np.concatenate([seg2, np.zeros(int(0.15*SR))])
    seg3 = meow(650, 0.45)
    hungry = np.concatenate([seg1, seg2, seg3])
    p = os.path.join(base_dir, "cat_sounds", "hungry_feeding_time.wav")
    save_wav(hungry, p); files.append(("hungry", p, analyze(hungry)))

    # happy: 轻柔喵叫 + 呼噜 (低频 ~25Hz)
    t_happy = np.arange(0, 1.6, 1/SR)
    purr = 0.12 * np.sin(2 * np.pi * 27 * t_happy)  # 呼噜 27Hz
    happy = meow(440, 1.2, vibrato=True)
    happy_pad = np.pad(happy, (int(0.15*SR), int(0.25*SR)), mode='constant')[:len(purr)]
    final_happy = happy_pad * 0.8 + purr
    p = os.path.join(base_dir, "cat_sounds", "happy_purring.wav")
    save_wav(final_happy, p); files.append(("happy", p, analyze(final_happy)))

    # angry: 嘶嘶声 + 低吼
    t_angry = np.arange(0, 1.2, 1/SR)
    hiss = 0.25 * np.random.randn(len(t_angry))  # 白噪声
    hiss_env = envelope(len(t_angry), attack=0.01, release=0.06)
    hiss = hiss * hiss_env
    growl = 0.35 * meow(170, 1.0, vibrato=False)
    growl_pad = np.pad(growl, (int(0.1*SR), len(t_angry)-int(0.1*SR)-len(growl)), mode='constant')
    final_angry = hiss + growl_pad
    p = os.path.join(base_dir, "cat_sounds", "angry_hissing.wav")
    save_wav(final_angry, p); files.append(("angry", p, analyze(final_angry)))

    # fear: 颤抖高音喵叫
    fear1 = meow(750, 0.3)
    fear2 = meow(800, 0.25)
    fear = np.concatenate([fear1, np.zeros(int(0.1*SR)), fear2])
    p = os.path.join(base_dir, "cat_sounds", "fear_shaky.wav")
    save_wav(fear, p); files.append(("fear", p, analyze(fear)))

    # seek_attention: 长喵叫, 重复
    sa1 = meow(500, 0.7)
    sa2 = meow(520, 0.7)
    sa = np.concatenate([sa1, np.zeros(int(0.2*SR)), sa2])
    p = os.path.join(base_dir, "cat_sounds", "seek_attention_petme.wav")
    save_wav(sa, p); files.append(("seek_attention", p, analyze(sa)))

    # pain: 尖锐短促喵叫
    pain1 = meow(950, 0.2)
    pain2 = meow(1020, 0.18)
    pain = np.concatenate([pain1, np.zeros(int(0.08*SR)), pain2])
    p = os.path.join(base_dir, "cat_sounds", "pain_injured.wav")
    save_wav(pain, p); files.append(("pain", p, analyze(pain)))

    # alert: 短促chatter
    segs = []
    for freq in [700, 720, 690]:
        s = meow(freq, 0.1)
        segs.append(np.concatenate([s, np.zeros(int(0.06*SR))]))
    segs.append(meow(710, 0.1))
    alert = np.concatenate(segs)
    p = os.path.join(base_dir, "cat_sounds", "alert_bird_watching.wav")
    save_wav(alert, p); files.append(("alert", p, analyze(alert)))

    # excited: 快速连续短喵 + 颤音高
    segs = []
    for freq in [680, 720, 760, 740]:
        dur = 0.18 if freq < 760 else 0.2
        s = meow(freq, dur)
        if freq != 740:
            s = np.concatenate([s, np.zeros(int(0.05*SR))])
        segs.append(s)
    excited = np.concatenate(segs)
    p = os.path.join(base_dir, "cat_sounds", "excited_treat_time.wav")
    save_wav(excited, p); files.append(("excited", p, analyze(excited)))

    # anxious: 断续呜咽 + 低频
    t_anxious = np.arange(0, 2.0, 1/SR)
    a1 = meow(380, 0.5) * 0.7
    a2 = meow(360, 0.6) * 0.8
    anxious = np.concatenate([a1, np.zeros(int(0.25*SR)), a2])
    anxious_pad = np.pad(anxious, (0, len(t_anxious)-len(anxious)), mode='constant')
    p = os.path.join(base_dir, "cat_sounds", "anxious_vet_visit.wav")
    save_wav(anxious_pad, p); files.append(("anxious", p, analyze(anxious_pad)))

    # content: 长时间呼噜
    t_content = np.arange(0, 2.5, 1/SR)
    content = 0.28 * np.sin(2 * np.pi * 26 * t_content) + 0.06 * np.sin(2 * np.pi * 52 * t_content)
    content_env = envelope(len(t_content), attack=0.15, release=0.25)
    content = content * content_env
    p = os.path.join(base_dir, "cat_sounds", "content_blanket.wav")
    save_wav(content, p); files.append(("content", p, analyze(content)))

    # frustrated: 喷气 + 重音喵
    t_frustr = np.arange(0, 1.5, 1/SR)
    spkitty = 0.3 * np.random.randn(len(t_frustr)) * np.exp(-np.arange(len(t_frustr))/1000)
    fr_meow = meow(520, 0.4) * 0.7
    # 填充到正确长度: 先 padding 然后放到 spkitty 上
    fr_before = int(0.25 * SR)
    fr_after = len(t_frustr) - len(fr_meow) - fr_before
    if fr_after < 0:
        fr_meow = fr_meow[:len(t_frustr) - fr_before]
        fr_after = 0
    fr_meow_pad = np.pad(fr_meow, (fr_before, fr_after), mode='constant')
    if len(fr_meow_pad) < len(t_frustr):
        fr_meow_pad = np.pad(fr_meow_pad, (0, len(t_frustr) - len(fr_meow_pad)))
    elif len(fr_meow_pad) > len(t_frustr):
        fr_meow_pad = fr_meow_pad[:len(t_frustr)]
    final_frustr = spkitty + fr_meow_pad
    p = os.path.join(base_dir, "cat_sounds", "frustrated_door_closed.wav")
    save_wav(final_frustr, p); files.append(("frustrated", p, analyze(final_frustr)))

    # lonely: 门口长喵
    lo1 = meow(500, 0.8)
    lo2 = meow(470, 0.9)
    lonely = np.concatenate([lo1, np.zeros(int(0.35*SR)), lo2])
    p = os.path.join(base_dir, "cat_sounds", "lonely_home_alone.wav")
    save_wav(lonely, p); files.append(("lonely", p, analyze(lonely)))

    # 保留原 test.wav (如果存在) 或生成默认文件
    test_default = meow(590, 1.5)
    p = os.path.join(base_dir, "cat_sounds", "test.wav")
    save_wav(test_default, p); files.append(("test", p, analyze(test_default)))

    return files


# ============ 狗叫数据集 (12个) ============
def gen_dog_dataset(base_dir):
    print("\n🐶 生成狗叫测试数据集 (12 个文件)...")
    print("=" * 60)
    files = []

    # hungry: 重复汪汪叫
    hungry = bark_sound(520, 1.2, n_barks=4)
    p = os.path.join(base_dir, "dog_sounds", "hungry_dinner_time.wav")
    save_wav(hungry, p); files.append(("hungry", p, analyze(hungry)))

    # happy: 摇尾伴随欢快短叫
    happy = bark_sound(560, 1.0, n_barks=6, pitch_drop=False)
    # 叠加呼吸声
    t = np.arange(0, len(happy)/SR, 1/SR)
    happy = happy + 0.05 * np.random.randn(len(happy)) * np.abs(np.sin(2*np.pi*3*t))
    p = os.path.join(base_dir, "dog_sounds", "happy_owner_home.wav")
    save_wav(happy, p); files.append(("happy", p, analyze(happy)))

    # angry: 深沉低吼 + 吠叫
    t_angry = np.arange(0, 2.0, 1/SR)
    growl = 0.4 * bark_sound(180, 0.9, n_barks=1, pitch_drop=True)
    bark_part = 0.9 * bark_sound(480, 1.0, n_barks=3)
    final_angry = np.pad(growl, (0, max(0, len(t_angry)-len(growl))))[:len(t_angry)]
    bark_pad = np.pad(bark_part, (max(0, len(t_angry)-len(bark_part)), 0))[:len(t_angry)]
    final_angry = 0.7 * final_angry + 0.8 * bark_pad
    p = os.path.join(base_dir, "dog_sounds", "angry_stranger.wav")
    save_wav(final_angry, p); files.append(("angry", p, analyze(final_angry)))

    # fear: 呜咽
    fear = 0.7 * bark_sound(350, 0.6, n_barks=2)
    p = os.path.join(base_dir, "dog_sounds", "fear_thunder.wav")
    save_wav(fear, p); files.append(("fear", p, analyze(fear)))

    # seek_attention: 爪子抓挠 + 哀求叫
    sa = bark_sound(430, 0.9, n_barks=3)
    p = os.path.join(base_dir, "dog_sounds", "seek_attention_walk.wav")
    save_wav(sa, p); files.append(("seek_attention", p, analyze(sa)))

    # pain: 尖叫
    pain = bark_sound(1200, 0.25, n_barks=2, pitch_drop=True)
    p = os.path.join(base_dir, "dog_sounds", "pain_stepped_on.wav")
    save_wav(pain, p); files.append(("pain", p, analyze(pain)))

    # alert: 吠叫 (连续3次+2次)
    a1 = bark_sound(500, 0.7, n_barks=3)
    a2 = bark_sound(520, 0.6, n_barks=2)
    alert = np.concatenate([a1, np.zeros(int(0.2*SR)), a2])
    p = os.path.join(base_dir, "dog_sounds", "alert_doorbell.wav")
    save_wav(alert, p); files.append(("alert", p, analyze(alert)))

    # excited: 跳跃汪叫 + 高音
    excited = bark_sound(620, 1.0, n_barks=8, pitch_drop=False)
    p = os.path.join(base_dir, "dog_sounds", "excited_leash_time.wav")
    save_wav(excited, p); files.append(("excited", p, analyze(excited)))

    # territorial: 深沉持续吠
    t1 = bark_sound(320, 1.0, n_barks=2)
    t2 = bark_sound(340, 0.7, n_barks=1)
    terr = np.concatenate([t1, np.zeros(int(0.3*SR)), t2])
    p = os.path.join(base_dir, "dog_sounds", "territorial_fence.wav")
    save_wav(terr, p); files.append(("territorial", p, analyze(terr)))

    # greeting: 轻快汪 + 摇尾喘气
    greeting = bark_sound(480, 0.6, n_barks=4)
    t = np.arange(0, len(greeting)/SR, 1/SR)
    breath = 0.07 * np.random.randn(len(greeting)) * np.abs(np.sin(2*np.pi*4*t))
    greeting = greeting + breath
    p = os.path.join(base_dir, "dog_sounds", "greeting_welcome.wav")
    save_wav(greeting, p); files.append(("greeting", p, analyze(greeting)))

    # anxious: 持续呜咽踱步
    x1 = bark_sound(360, 1.0, n_barks=2)
    x2 = bark_sound(340, 1.1, n_barks=2)
    anxious = np.concatenate([x1, np.zeros(int(0.2*SR)), x2])
    p = os.path.join(base_dir, "dog_sounds", "anxious_alone.wav")
    save_wav(anxious, p); files.append(("anxious", p, analyze(anxious)))

    # playful: 玩耍吠 (断断续续)
    p1 = bark_sound(550, 0.9, n_barks=5)
    p2 = bark_sound(580, 0.5, n_barks=3)
    playful = np.concatenate([p1, np.zeros(int(0.25*SR)), p2])
    p = os.path.join(base_dir, "dog_sounds", "playful_fetch.wav")
    save_wav(playful, p); files.append(("playful", p, analyze(playful)))

    return files


# ============ 人类语音数据集 (6个) ============
def gen_human_dataset(base_dir):
    print("\n👤 生成人类语音测试数据集 (6 个文件)...")
    print("=" * 60)
    files = []

    sentences = [
        ("过来吃饭啦", 1.6, "command_feeding"),
        ("小猫咪过来", 1.4, "call_cat_name"),
        ("狗狗坐下握手", 1.8, "command_trick"),
        ("你今天开心吗", 1.5, "question_mood"),
        ("乖乖睡觉去吧", 1.6, "soothing_sleep"),
        ("走我们去散步", 1.4, "command_walk"),
    ]
    for text, dur, tag in sentences:
        audio = human_speech_synth(text, dur)
        p = os.path.join(base_dir, "human_sounds", f"{tag}.wav")
        save_wav(audio, p); files.append((tag, p, analyze(audio)))
    return files


# ============ 主程序 ============
def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    print(f"📁 数据目录: {base_dir}")

    # 创建目录
    for d in ["cat_sounds", "dog_sounds", "human_sounds"]:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    cat_files   = gen_cat_dataset(base_dir)
    dog_files   = gen_dog_dataset(base_dir)
    human_files = gen_human_dataset(base_dir)

    # 汇总
    print("\n" + "=" * 60)
    print("📊 数据集汇总")
    print("=" * 60)
    all_files = [("cat", cat_files), ("dog", dog_files), ("human", human_files)]
    grand_total = 0
    for pet_name, flist in all_files:
        dur_total = 0.0
        size_total = 0
        print(f"\n  [{pet_name}] {len(flist)} 个文件:")
        print(f"  {'文件名':50s}  {'情绪':16s} {'时长':>6s} {'主频':>8s}  Top3频率")
        print("  " + "-"*110)
        for emo, path, (peak, rms, mf, top3) in flist:
            dur = len(wavfile.read(path)[1]) / SR
            size = os.path.getsize(path) / 1024
            dur_total += dur
            size_total += size
            top3s = f"{top3[0]:.0f}/{top3[1]:.0f}/{top3[2]:.0f}"
            print(f"  {os.path.basename(path):50s} {emo:16s} {dur:5.1f}s {mf:7.0f}Hz  {top3s}")
            grand_total += 1
        print(f"  小计: {dur_total:.1f}秒, {size_total/1024:.1f}MB")
    print(f"\n  总计: {grand_total} 个文件")

if __name__ == "__main__":
    main()
