#!/usr/bin/env python3
"""
模型评估模块 scripts/evaluate_models.py
==========================================
- 自动扫描 checkpoints/ 下的 LLM LoRA / Whisper 模型
- 使用 data/raw/ 下真实合成的猫 / 狗 / 人 语音数据 (已用 generate_test_audio.py 生成)
- 三层评估维度:
    1. Whisper ASR 评估:       情绪关键词命中率、文本分布多样性
    2. LLM LoRA 评估:         情绪分类准确率、需求/建议语义匹配度
    3. 端到端 pet_to_text:    Top1/Top3 情绪命中 (RAG + LLM 联合)
- 输出评估 JSON + 人类可读 Markdown 报告 (写入 README 评估章节)
"""
import os
import sys
import json
import time
import re
import glob
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field

# -------- 路径 --------
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

DATA_DIR  = BASE_DIR / "data" / "raw"
CKPT_DIR  = BASE_DIR / "checkpoints"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

REPORT_JSON = OUT_DIR / "model_evaluation_report.json"
REPORT_MD   = OUT_DIR / "model_evaluation_report.md"

# -------- 20 种情绪的中英文关键词池 (用于 Whisper/LLM 结果关键词匹配) --------
EMOTION_KEYWORDS = {
    # 20 情绪 -> (cn_keywords, en_keywords)
    "hungry":         (["饿", "喂饭", "饭点", "吃", "喂食", "粮食"], ["hungry", "feed", "eat", "dinner"]),
    "happy":          (["开心", "高兴", "愉悦", "满足", "舒", "呼噜"], ["happy", "joy", "purr", "satisfied"]),
    "angry":          (["生气", "愤怒", "怒", "低吼", "嘶", "警告"], ["angry", "hiss", "growl", "warning"]),
    "fear":           (["害怕", "恐惧", "抖", "呜", "雷声", "怕"], ["fear", "afraid", "scared", "shake"]),
    "seek_attention": (["求抚摸", "求关注", "摸", "玩", "陪", "吸引"], ["attention", "pet me", "play", "companion"]),
    "content":        (["满足", "放松", "舒服", "安逸", "安详"], ["content", "relaxed", "peaceful", "cozy"]),
    "pain":           (["痛", "受伤", "惨叫", "尖叫", "病"], ["pain", "hurt", "injured", "sick"]),
    "alert":          (["警惕", "警戒", "警报", "察觉", "陌生", "门铃"], ["alert", "watch", "stranger", "doorbell"]),
    "playful":        (["玩", "玩耍", "嬉闹", "游戏", "接飞盘", "逗"], ["playful", "fetch", "play", "toy"]),
    "sad":            (["悲伤", "难过", "哭", "低落"], ["sad", "down", "cry", "depressed"]),
    "curious":        (["好奇", "探索", "打量", "盯", "新东西"], ["curious", "explore", "investigate"]),
    "lonely":         (["孤独", "孤单", "一人", "没人陪", "等主人"], ["lonely", "alone", "miss", "owner"]),
    "anxious":        (["焦虑", "不安", "紧张", "踱步", "担心", "慌"], ["anxious", "nervous", "tense", "worry"]),
    "excited":        (["兴奋", "激动", "蹦", "跳", "雀跃", "奖励"], ["excited", "thrilled", "bounce", "treat"]),
    "frustrated":     (["挫败", "沮丧", "受挫", "烦躁", "阻挠", "没成"], ["frustrated", "defeated", "blocked", "annoyed"]),
    "relaxed":        (["放松", "从容", "安逸", "悠闲", "懒懒"], ["relaxed", "calm", "at ease", "leisurely"]),
    "territorial":    (["领地", "护家", "护食", "地盘", "闯入"], ["territorial", "domain", "protect", "intruder"]),
    "greeting":       (["问候", "欢迎", "迎接", "主人回来", "打招呼"], ["greeting", "welcome", "hello", "owner home"]),
    "confused":       (["困惑", "懵", "不懂", "疑惑", "不确定"], ["confused", "puzzled", "doubt", "uncertain"]),
    "jealous":        (["嫉妒", "吃醋", "争宠", "不乐意", "排斥"], ["jealous", "envy", "vying", "rivals"]),
}

EMOTION_CN = {
    "hungry":"饥饿", "happy":"开心", "angry":"愤怒", "fear":"恐惧",
    "seek_attention":"求关注", "content":"满足", "pain":"疼痛", "alert":"警惕",
    "playful":"玩耍", "sad":"悲伤", "curious":"好奇", "lonely":"孤独",
    "anxious":"焦虑", "excited":"兴奋", "frustrated":"挫败", "relaxed":"放松",
    "territorial":"领地意识", "greeting":"问候", "confused":"困惑", "jealous":"嫉妒",
}

# -------- 自动发现模型 --------
def discover_models():
    found = {"llm": [], "whisper": []}
    # LLM LoRA: 包含 adapter_config.json 且 base_model 含 TinyLlama/Qwen 等
    for ac in sorted(CKPT_DIR.rglob("adapter_config.json")):
        try:
            cfg = json.loads(ac.read_text(encoding="utf-8"))
        except Exception:
            continue
        base = str(cfg.get("base_model_name_or_path", "")).lower()
        p = str(ac.parent)
        if "whisper" in base or "whisper" in p.lower():
            found["whisper"].append({"path": p, "base": base, "name": ac.parent.name})
        else:
            found["llm"].append({"path": p, "base": base, "name": "/".join(ac.parent.relative_to(CKPT_DIR).parts)})
    # 取最新/最大的 LLM checkpoint (按 step)
    def _step(name):
        m = re.search(r"checkpoint-(\d+)", name)
        return int(m.group(1)) if m else (999999 if name == "final" else 0)
    found["llm"].sort(key=lambda m: -_step(m["name"]))
    found["whisper"].sort(key=lambda m: -_step(m["name"]))
    return found

# -------- 训练曲线读取 (trainer_state.json) --------
def read_training_state(model_path):
    state_p = Path(model_path) / "trainer_state.json"
    if not state_p.exists():
        return None
    try:
        s = json.loads(state_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    log_history = s.get("log_history", [])
    # 取最后一条 train / eval
    last_train = next((l for l in reversed(log_history) if "loss" in l), {})
    last_eval  = next((l for l in reversed(log_history) if "eval_loss" in l), {})
    steps = s.get("global_step", "-")
    epochs = s.get("epoch", "-")
    return {
        "global_step": steps,
        "epoch": epochs,
        "last_train_loss": last_train.get("loss"),
        "last_train_lr":  last_train.get("learning_rate"),
        "last_eval_loss":  last_eval.get("eval_loss"),
        "last_eval_runtime_s": last_eval.get("eval_runtime"),
        "best_model_checkpoint": s.get("best_model_checkpoint"),
    }

# -------- 情绪关键词命中判断 --------
def emotion_from_keywords(text, true_emotion=None):
    """根据文本关键词返回最可能的情绪 + true_emotion 是否命中 TopK"""
    scores = {}
    if not text:
        return {}, False, False
    t = text.lower()
    for emo, (cn_ks, en_ks) in EMOTION_KEYWORDS.items():
        hits = 0
        for k in cn_ks:
            if k and k in text: hits += 1
        for k in en_ks:
            if k and k in t:    hits += 1
        scores[emo] = hits
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_emos = [e for e, s in ranked if s > 0][:3]
    # 若关键词完全没命中, 也按最高 3 个给出
    if not top_emos:
        top_emos = [e for e, _ in ranked[:3]]
    top1_hit = (true_emotion == top_emos[0]) if true_emotion else False
    top3_hit = (true_emotion in top_emos)   if true_emotion else False
    return dict(ranked), top1_hit, top3_hit, top_emos

# -------- 数据集: 从文件名标签自动构造 --------
def build_ground_truth():
    """扫描 data/raw 下 cat/dog wav, 用文件名第一段下划线前缀作为情绪标签"""
    gt = []  # {path, pet, breed, emotion_cn, emotion_en, expected_asr_keywords}
    for pet_dir, pet in [("cat_sounds", "cat"), ("dog_sounds", "dog")]:
        d = DATA_DIR / pet_dir
        if not d.exists(): continue
        for wav in sorted(d.glob("*.wav")):
            stem = wav.stem
            emo_en = stem.split("_")[0]  # 例: hungry_feeding_time -> hungry
            if emo_en not in EMOTION_KEYWORDS:
                # test / 其它 -> 跳过正式评估, 只保留 N/A
                continue
            # 用文件名场景暗示生成 expected_asr_keywords (期望 Whisper 里有的词)
            scene = stem.replace("_", " ")
            gt.append({
                "path": str(wav),
                "pet": pet,
                "breed": "通用",
                "emotion_en": emo_en,
                "emotion_cn": EMOTION_CN.get(emo_en, emo_en),
                "scene": scene,
            })
    return gt

# =========================================================
# 评估 1: Whisper ASR
# =========================================================
@dataclass
class WhisperEvalResult:
    model_path: str
    base_model: str
    training_state: dict = None
    total_samples: int = 0
    # ASR 输出多样性
    unique_texts: int = 0
    top_texts: list = field(default_factory=list)
    empty_outputs: int = 0
    fallback_outputs: int = 0  # ≤2字, 嗚/汪/呜 这种
    # 情绪关键词命中 (ASR文本 -> 情绪关键词 -> 情绪)
    top1_hit: int = 0
    top3_hit: int = 0
    per_emotion: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)  # 每样本明细
    elapsed_s: float = 0.0

# =========================================================
# 评估 2: LLM 情绪/需求匹配
# =========================================================
@dataclass
class LLMEvalResult:
    model_path: str
    base_model: str
    training_state: dict = None
    total_samples: int = 0
    # 情绪分类: LLM 文本 -> 情绪关键词 -> 判断
    emotion_top1_hit: int = 0
    emotion_top3_hit: int = 0
    # 需求/建议: 文本包含 "建议/应该/可以" 等词的比例
    advice_covered: int = 0
    meaning_covered: int = 0
    avg_output_len: float = 0.0
    per_emotion: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    elapsed_s: float = 0.0

# =========================================================
# 评估 3: 端到端 pet_to_text
# =========================================================
@dataclass
class E2EEvalResult:
    total_samples: int = 0
    rag_top1_hit: int = 0
    rag_top3_hit: int = 0
    llm_top1_hit: int = 0
    llm_top3_hit: int = 0
    per_emotion: dict = field(default_factory=dict)
    samples: list = field(default_factory=list)
    elapsed_s: float = 0.0


def evaluate_whisper(whisper_model_info, gt):
    from src.models.whisper_finetune import WhisperFineTuner, WhisperFineTuneConfig
    path = whisper_model_info["path"]
    print(f"\n{'='*72}")
    print(f"🎙️  [评估 1/3] Whisper ASR: {path}")
    print(f"{'='*72}")
    t0 = time.time()
    res = WhisperEvalResult(model_path=path, base_model=whisper_model_info["base"])
    res.training_state = read_training_state(path)
    print(f"   训练状态: {json.dumps(res.training_state, ensure_ascii=False)}")

    # 加载: 先构造 WhisperFineTuner(config), 再 load_finetuned
    ft = None
    try:
        cfg = WhisperFineTuneConfig()
        ft = WhisperFineTuner(cfg)  # 先加载基础 whisper-tiny (网络不可用则会内部降级)
        if path and os.path.isdir(path):
            ft.load_finetuned(path)
        if not ft.is_ready:
            print(f"   ⚠️  Whisper 基础/微调模型均未就绪，使用 fallback (纯基础加载)")
    except Exception as e:
        print(f"   ❌ Whisper 加载异常: {e}")
        try:
            cfg = WhisperFineTuneConfig()
            ft = WhisperFineTuner(cfg)
        except Exception as e2:
            print(f"   ❌ 基础 Whisper 也加载失败: {e2} —— 将空跑")
            ft = None

    FALLBACK_CHARS = {"嗚", "呜", "汪", "吠", "叫", "啊", "嗯", "哦", "", "哎", "哼"}
    emo_stats = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})

    for i, item in enumerate(gt, 1):
        try:
            text = ft.inference(item["path"])
        except Exception as e:
            text = f"[ERROR] {e}"
        text_clean = text.strip()
        unique, top1, top3, top3_emos = emotion_from_keywords(text_clean, item["emotion_en"])

        empty = len(text_clean) == 0
        fallback = (len(text_clean) <= 2 and (not text_clean or text_clean in FALLBACK_CHARS))
        if empty:    res.empty_outputs += 1
        if fallback: res.fallback_outputs += 1

        emo_stats[item["emotion_en"]]["n"] += 1
        if top1: emo_stats[item["emotion_en"]]["top1"] += 1
        if top3: emo_stats[item["emotion_en"]]["top3"] += 1
        res.top1_hit += int(top1)
        res.top3_hit += int(top3)

        sample = {
            "idx": i,
            "pet": item["pet"],
            "scene": item["scene"],
            "emotion": f"{item['emotion_en']}({item['emotion_cn']})",
            "whisper_text": text_clean,
            "is_empty": empty,
            "is_fallback": fallback,
            "pred_top3_emotions": top3_emos,
            "top1_hit": top1,
            "top3_hit": top3,
        }
        res.samples.append(sample)
        mark = "✅" if top1 else ("🔶" if top3 else "❌")
        print(f"   [{i:3d}/{len(gt)}] {mark} {item['pet']:3s} {item['emotion_cn']:4s} "
              f"Whisper='{text_clean}' -> Top3={top3_emos}")

    # 汇总
    res.total_samples = len(gt)
    text_counter = Counter(s["whisper_text"] for s in res.samples)
    res.unique_texts = len(text_counter)
    res.top_texts = text_counter.most_common(8)
    for k, v in emo_stats.items():
        v["top1_acc"] = round(v["top1"]/v["n"], 3) if v["n"] else 0
        v["top3_acc"] = round(v["top3"]/v["n"], 3) if v["n"] else 0
        res.per_emotion[k] = v
    res.elapsed_s = round(time.time()-t0, 2)
    print(f"   ⏱️  {res.elapsed_s}s  "
          f"Top1={res.top1_hit}/{res.total_samples} ({res.top1_hit/res.total_samples*100:.1f}%)  "
          f"Top3={res.top3_hit}/{res.total_samples} ({res.top3_hit/res.total_samples*100:.1f}%)  "
          f"fallback={res.fallback_outputs}  empty={res.empty_outputs}")
    return res


def evaluate_llm(llm_model_info, gt):
    from src.models.llm_finetune import LLMFineTuner, LLMFineTuneConfig
    path = llm_model_info["path"]
    print(f"\n{'='*72}")
    print(f"🧠  [评估 2/3] LLM LoRA: {path}")
    print(f"{'='*72}")
    t0 = time.time()
    res = LLMEvalResult(model_path=path, base_model=llm_model_info["base"])
    res.training_state = read_training_state(path)
    print(f"   训练状态: {json.dumps(res.training_state, ensure_ascii=False)}")

    llm = None
    try:
        cfg = LLMFineTuneConfig()
        llm = LLMFineTuner(cfg, adapter_path=path)
        if not llm.is_ready:
            print(f"   ⚠️  LLM 加载后 not ready —— 尝试纯 config 加载")
    except Exception as e:
        print(f"   ❌ LLM 加载失败: {e}")
        try:
            cfg = LLMFineTuneConfig()
            llm = LLMFineTuner(cfg)
        except Exception as e2:
            print(f"   ❌ 基础 LLM 也加载失败: {e2}")
            llm = None

    emo_stats = defaultdict(lambda: {"n":0, "emo_top1":0, "emo_top3":0, "advice":0, "meaning":0})
    total_len = 0

    # 给 LLM 喂一个模拟的 "声音描述+RAG上下文" 提示
    for i, item in enumerate(gt, 1):
        # 模拟: Whisper 输出 + RAG context
        sys_prompt = "你是一个宠物行为学专家。"
        prompt = (
            f"宠物类型: {item['pet']}\n"
            f"品种: {item['breed']}\n"
            f"声音描述: 发出{EMOTION_CN.get(item['emotion_en'],'情绪')}的叫声，场景: {item['scene']}\n"
            f"知识库参考:\n"
            f"  - [{item['emotion_en']}] 声音: 典型{EMOTION_CN[item['emotion_en']]}声音\n"
            f"  - 含义: 宠物正在表达{EMOTION_CN[item['emotion_en']]}的情绪\n"
            f"  - 建议: 请根据情绪做出适当反应\n\n"
            "请输出以下格式:\n"
            "1. 情绪判断: <情绪>\n"
            "2. 具体含义: <含义>\n"
            "3. 行动建议: <建议>\n"
        )
        try:
            if llm:
                out = llm.inference(prompt, sys_prompt)
            else:
                out = ""
        except Exception as e:
            out = f"[ERROR] {e}"

        _, top1, top3, top3_emos = emotion_from_keywords(out, item["emotion_en"])
        has_advice  = any(w in out for w in ["建议", "应该", "可以", "需", "尝试"])
        has_meaning = any(w in out for w in ["含义", "代表", "表达", "说明", "感到", "情绪", "需求"])
        total_len += len(out)

        emo_stats[item["emotion_en"]]["n"] += 1
        if top1:      emo_stats[item["emotion_en"]]["emo_top1"] += 1
        if top3:      emo_stats[item["emotion_en"]]["emo_top3"] += 1
        if has_advice:  emo_stats[item["emotion_en"]]["advice"] += 1
        if has_meaning: emo_stats[item["emotion_en"]]["meaning"] += 1
        res.emotion_top1_hit += int(top1)
        res.emotion_top3_hit += int(top3)
        res.advice_covered  += int(has_advice)
        res.meaning_covered += int(has_meaning)

        mark = "✅" if top1 else ("🔶" if top3 else "❌")
        preview = out.replace("\n", " ")[:80]
        print(f"   [{i:3d}/{len(gt)}] {mark} {item['pet']:3s} {item['emotion_cn']:4s} "
              f"Top3={top3_emos} 含义={has_meaning} 建议={has_advice} "
              f"| {preview}...")
        res.samples.append({
            "idx": i,
            "pet": item["pet"],
            "scene": item["scene"],
            "emotion": f"{item['emotion_en']}({item['emotion_cn']})",
            "pred_top3_emotions": top3_emos,
            "has_meaning": has_meaning,
            "has_advice": has_advice,
            "output_len": len(out),
            "output_preview": out[:200],
            "emo_top1_hit": top1,
            "emo_top3_hit": top3,
        })

    res.total_samples = len(gt)
    res.avg_output_len = round(total_len / max(1, res.total_samples), 1)
    for k, v in emo_stats.items():
        v["emo_top1_acc"] = round(v["emo_top1"]/v["n"],3) if v["n"] else 0
        v["emo_top3_acc"] = round(v["emo_top3"]/v["n"],3) if v["n"] else 0
        v["advice_rate"]  = round(v["advice"]/v["n"],3) if v["n"] else 0
        v["meaning_rate"] = round(v["meaning"]/v["n"],3) if v["n"] else 0
        res.per_emotion[k] = v
    res.elapsed_s = round(time.time()-t0, 2)
    print(f"   ⏱️  {res.elapsed_s}s  "
          f"情绪Top1={res.emotion_top1_hit}/{res.total_samples} ({res.emotion_top1_hit/res.total_samples*100:.1f}%)  "
          f"情绪Top3={res.emotion_top3_hit}/{res.total_samples} ({res.emotion_top3_hit/res.total_samples*100:.1f}%)  "
          f"含义覆盖率={res.meaning_covered/res.total_samples*100:.0f}%  "
          f"建议覆盖率={res.advice_covered/res.total_samples*100:.0f}%  "
          f"平均字数={res.avg_output_len}")
    return res


def evaluate_e2e(whisper_eval_res, llm_eval_res, gt,
                 whisper_model_info, llm_model_info):
    """端到端 E2E: 用 TranslationPipeline 跑 pet_sound_to_text"""
    from src.inference.pipeline import TranslationPipeline
    print(f"\n{'='*72}")
    print(f"🔗  [评估 3/3] 端到端 pet_to_text (Whisper+RAG+LLM)")
    print(f"{'='*72}")
    t0 = time.time()
    res = E2EEvalResult()
    try:
        pipe = TranslationPipeline(
            whisper_model_path=whisper_model_info["path"],
            llm_adapter_path=llm_model_info["path"],
            use_rag=True,
            use_agent=False,
        )
    except Exception as e:
        print(f"   ❌ Pipeline init 失败: {e}")
        return res

    emo_stats = defaultdict(lambda: {"n":0, "rag_t1":0, "rag_t3":0, "llm_t1":0, "llm_t3":0})
    for i, item in enumerate(gt, 1):
        try:
            out = pipe.pet_sound_to_text(item["path"], pet_type=item["pet"], breed=item["breed"])
        except Exception as e:
            out = {"rag_details": [], "analysis": f"[ERROR] {e}"}
        # RAG 侧情绪 Top1/Top3
        rag_emos = []
        for r in out.get("rag_details", [])[:3]:
            e = r.get("metadata", {}).get("emotion", "")
            if e: rag_emos.append(e)
        rag_t1 = rag_emos[0] == item["emotion_en"] if rag_emos else False
        rag_t3 = item["emotion_en"] in rag_emos       if rag_emos else False

        # LLM 侧情绪 Top1/Top3
        analysis = out.get("analysis", "")
        _, llm_t1, llm_t3, llm_emos = emotion_from_keywords(analysis, item["emotion_en"])

        emo_stats[item["emotion_en"]]["n"] += 1
        if rag_t1: emo_stats[item["emotion_en"]]["rag_t1"] += 1
        if rag_t3: emo_stats[item["emotion_en"]]["rag_t3"] += 1
        if llm_t1: emo_stats[item["emotion_en"]]["llm_t1"] += 1
        if llm_t3: emo_stats[item["emotion_en"]]["llm_t3"] += 1
        res.rag_top1_hit += int(rag_t1)
        res.rag_top3_hit += int(rag_t3)
        res.llm_top1_hit += int(llm_t1)
        res.llm_top3_hit += int(llm_t3)

        mark_rag = "✅" if rag_t1 else ("🔶" if rag_t3 else "❌")
        mark_llm = "✅" if llm_t1 else ("🔶" if llm_t3 else "❌")
        whisper_out = out.get("whisper_transcription", "")
        analysis_pv = analysis.replace("\n"," ")[:60]
        print(f"   [{i:3d}/{len(gt)}] {item['pet']:3s} {item['emotion_cn']:4s} "
              f"RAG{mark_rag}{rag_emos[:2]}  LLM{mark_llm}{llm_emos[:2]}  "
              f"Whisper='{whisper_out}'")
        res.samples.append({
            "idx": i,
            "pet": item["pet"],
            "scene": item["scene"],
            "emotion": f"{item['emotion_en']}({item['emotion_cn']})",
            "whisper_text": whisper_out,
            "rag_top3": rag_emos,
            "llm_top3": llm_emos,
            "rag_top1_hit": rag_t1, "rag_top3_hit": rag_t3,
            "llm_top1_hit": llm_t1, "llm_top3_hit": llm_t3,
            "analysis_preview": analysis_pv,
        })

    res.total_samples = len(gt)
    for k, v in emo_stats.items():
        v["rag_t1_acc"] = round(v["rag_t1"]/v["n"],3) if v["n"] else 0
        v["rag_t3_acc"] = round(v["rag_t3"]/v["n"],3) if v["n"] else 0
        v["llm_t1_acc"] = round(v["llm_t1"]/v["n"],3) if v["n"] else 0
        v["llm_t3_acc"] = round(v["llm_t3"]/v["n"],3) if v["n"] else 0
        res.per_emotion[k] = v
    res.elapsed_s = round(time.time()-t0, 2)
    pct = lambda a,b: f"{a}/{b} ({a/b*100:.1f}%)" if b else "0"
    print(f"   ⏱️  {res.elapsed_s}s  "
          f"RAG Top1={pct(res.rag_top1_hit, res.total_samples)}  "
          f"RAG Top3={pct(res.rag_top3_hit, res.total_samples)}  "
          f"LLM Top1={pct(res.llm_top1_hit, res.total_samples)}  "
          f"LLM Top3={pct(res.llm_top3_hit, res.total_samples)}")
    return res


# -------- 报告生成 --------
def build_improvements(w_res: WhisperEvalResult, l_res: LLMEvalResult, e_res: E2EEvalResult):
    """基于评估数据生成可执行的改进建议"""
    imp = []
    N = max(1, w_res.total_samples)

    # Whisper 问题
    if w_res.fallback_outputs / N >= 0.5:
        imp.append({
            "level": "P0",
            "model": "Whisper",
            "problem": f"Whisper 退化率 {w_res.fallback_outputs/N*100:.0f}% —— 超过一半的样本被识别为 ≤2 字的 fallback (嗚/汪/呜)",
            "root": "微调数据不足 / 猫叫狗叫不在 openai/whisper-tiny 中文训练分布内，模型把非人声元音当成单字 fallback",
            "fix": [
                "① 准备 2000+ 条真实宠物叫声 + 分类标签 (hungry/happy/...) 重新微调 Whisper",
                "② 任务改为「多标签声音事件分类 (Audio Classification)」，不再走纯中文 ASR：用 AST / PANNs / AudioSet 预训练模型",
                "③ 若必须保留 ASR，将 Whisper 词表扩展为 20 种情绪 token + 声纹 token 映射",
            ]
        })
    if w_res.top3_hit / N < 0.4:
        imp.append({
            "level": "P0", "model": "Whisper",
            "problem": f"Whisper 文本→情绪关键词 Top3 命中率仅 {w_res.top3_hit/N*100:.0f}%",
            "root": "ASR 输出和情绪完全不相关，是噪声；下游关键词匹配等于猜",
            "fix": ["① 改造 Whisper fine-tune 任务为 情绪分类，输出 emotion logits",
                    "② 或前端加一个小 CNN 做基频+频谱情绪分类，独立决策"],
        })
    empty_rate = w_res.empty_outputs / N
    if empty_rate >= 0.1:
        imp.append({
            "level": "P1", "model": "Whisper",
            "problem": f"Whisper 空输出 {w_res.empty_outputs}/{N} ({empty_rate*100:.0f}%)",
            "root": "部分合成音频基频过高/过低，或者能量极低(如 content 呼噜 26Hz)，超出 Whisper 静音门限",
            "fix": ["① content/pain 等极低频极高频样本加 VAD 能量归一化 + 时长拉伸",
                    "② 推理前自动做 AGC (自动增益) + 高通滤波"],
        })

    # LLM 问题
    if l_res.total_samples:
        if l_res.emotion_top1_hit / l_res.total_samples < 0.6:
            imp.append({
                "level": "P1", "model": "LLM LoRA",
                "problem": f"LLM 情绪分类 Top1={l_res.emotion_top1_hit/l_res.total_samples*100:.0f}% < 60%",
                "root": "LoRA 训练集情绪标注可能偏少 / 输出格式不统一，关键词匹配法容易漏",
                "fix": ["① 扩展训练集: 每个情绪 ≥100 条 QA 对 (SFT)",
                        "② 让 LLM 严格输出 JSON: {emotion_en, meaning_cn, advice_cn} 然后做精确匹配",
                        "③ checkpoint-300 可能过拟合 / 欠拟合: 尝试 200/250 对比评估"],
            })
        meaning_rate = l_res.meaning_covered / l_res.total_samples
        advice_rate  = l_res.advice_covered  / l_res.total_samples
        if meaning_rate < 0.8:
            imp.append({
                "level": "P2", "model": "LLM LoRA",
                "problem": f"LLM 含义覆盖率仅 {meaning_rate*100:.0f}% (需 ≥80%)",
                "root": "训练集中缺乏 \"表达什么情绪/需求\" 的专门监督",
                "fix": ["在 SFT 数据中强制加入 \"具体含义:\" 字段模板"],
            })
        if advice_rate < 0.7:
            imp.append({
                "level": "P2", "model": "LLM LoRA",
                "problem": f"LLM 行动建议覆盖率仅 {advice_rate*100:.0f}% (需 ≥70%)",
                "root": "同上，建议字段模板缺失",
                "fix": ["在 SFT 数据中强制加入 \"行动建议:\" 字段模板"],
            })

    # E2E 问题
    if e_res.total_samples:
        if e_res.rag_top1_hit / e_res.total_samples < 0.35:
            imp.append({
                "level": "P1", "model": "RAG (Embedding)",
                "problem": f"RAG Top1 命中率 {e_res.rag_top1_hit/e_res.total_samples*100:.0f}% < 35%",
                "root": "Whisper 文本是乱码/噪声，Embedding 后和知识库中文描述差距大；知识库里 cat/dog 的声音描述以 \"急促短喵\" 等中文为主，ASR 出不来就匹配不上",
                "fix": ["① RAG 查询改为直接用 音频特征 (Audio Embedding, e.g. CLAP) 而不是文本",
                        "② 知识库中每个 entry 同时保存 多个 ASR 伪标签的同义词变体 (嗚/汪/喵/hungry...)",
                        "③ 查询用 pet_type+breed+场景词 重加权，而不是纯 ASR 文本"],
            })
        if e_res.llm_top3_hit / e_res.total_samples < 0.5:
            imp.append({
                "level": "P1", "model": "LLM+RAG 协同",
                "problem": f"E2E LLM 情绪 Top3 命中率 {e_res.llm_top3_hit/e_res.total_samples*100:.0f}%",
                "root": "上游 Whisper 识别噪声 + RAG 未命中 导致 LLM 输入完全是错的，是典型的 \"Garbage In Garbage Out\"",
                "fix": ["① 最优先修复 Whisper (改为音频分类，而不是 ASR 识别文字)",
                        "② 当 Whisper 文本≤2字时，直接跳过 ASR→RAG 文本路径，改为用 pet_type+breed+音频基频/能量/RMS 规则检索情绪",
                        "③ Pipeline 新增一层：声学特征→情绪 规则分类器 (决策树/lightGBM)，与 RAG+LLM 加权 ensemble"],
            })
    return imp


def build_markdown_report(found_models, whisper_res, llm_res, e2e_res, improvements):
    md = []
    md.append(f"# 🧪 模型评估报告 — 宠物声音翻译系统")
    md.append(f"\n> 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"> 自动扫描 `checkpoints/` 发现模型: **LLM={len(found_models['llm'])} 个, Whisper={len(found_models['whisper'])} 个**")

    # 1. 模型概览
    md.append(f"\n## 1. 已发现训练模型")
    for mtype, list_m in [("LLM LoRA", found_models["llm"]), ("Whisper LoRA", found_models["whisper"])]:
        md.append(f"\n### {mtype}")
        md.append("| # | 名称 | 路径 | 基座模型 |")
        md.append("|---|---|---|---|")
        for i, m in enumerate(list_m, 1):
            md.append(f"| {i} | {m['name']} | `{Path(m['path']).relative_to(BASE_DIR)}` | `{m['base']}` |")

    # 2. Whisper
    N = max(1, whisper_res.total_samples)
    ts = whisper_res.training_state or {}
    md.append(f"\n## 2. 🎙️ Whisper ASR 评估 — `{Path(whisper_res.model_path).relative_to(BASE_DIR)}`")
    md.append(f"\n### 2.1 训练状态")
    md.append(f"- Global step: **{ts.get('global_step','-')}** / Epoch: **{ts.get('epoch','-')}**")
    md.append(f"- Last train loss: `{ts.get('last_train_loss','-')}`  LR: `{ts.get('last_train_lr','-')}`")
    md.append(f"- Last eval loss: `{ts.get('last_eval_loss','-')}`  (eval time {ts.get('last_eval_runtime_s','-')}s)")
    md.append(f"- Best checkpoint: `{ts.get('best_model_checkpoint','-')}`")

    md.append(f"\n### 2.2 核心指标 (N={whisper_res.total_samples})")
    md.append(f"| 指标 | 值 | 说明 |")
    md.append(f"|---|---|---|")
    md.append(f"| 推理耗时 | `{whisper_res.elapsed_s}s`  ({whisper_res.elapsed_s/N:.2f}s/条) | 含 Mel+generate 全流程 |")
    md.append(f"| Top1 情绪关键词命中 | **{whisper_res.top1_hit}/{N} ({whisper_res.top1_hit/N*100:.1f}%)** | 关键词法第1情绪 = 真实情绪 |")
    md.append(f"| Top3 情绪关键词命中 | **{whisper_res.top3_hit}/{N} ({whisper_res.top3_hit/N*100:.1f}%)** | 前3候选包含真实情绪 |")
    md.append(f"| 独立输出文本数 | {whisper_res.unique_texts}/{N} | 衡量识别多样性 (越低越退化) |")
    md.append(f"| 空输出 | {whisper_res.empty_outputs} ({whisper_res.empty_outputs/N*100:.0f}%) | 完全没有输出 |")
    md.append(f"| Fallback 输出 (≤2字) | **{whisper_res.fallback_outputs} ({whisper_res.fallback_outputs/N*100:.0f}%)** | 嗚/汪/呜 等退化结果 |")

    md.append(f"\n### 2.3 Whisper 输出 Top 分布")
    md.append("```")
    for text, cnt in whisper_res.top_texts:
        md.append(f"  '{text}' × {cnt}  ({cnt/N*100:.0f}%)")
    md.append("```")

    md.append(f"\n### 2.4 Whisper 逐样本明细 (预期 vs 真实 vs 命中)")
    md.append("| # | 宠物 | 场景/真实情绪 | 🎯 Whisper 真实输出 | 预期情绪Top3 | Top1命中 | Top3命中 |")
    md.append("|---|---|---|---|---|---|---|")
    for s in whisper_res.samples:
        md.append(f"| {s['idx']} | {s['pet']} | {s['scene']}<br>`{s['emotion']}` | `{s['whisper_text']}` | {s['pred_top3_emotions']} | {'✅' if s['top1_hit'] else '❌'} | {'✅' if s['top3_hit'] else '❌'} |")

    md.append(f"\n### 2.5 分情绪 Whisper 准确率")
    md.append("| 情绪(中) | 情绪(en) | 样本数 | Top1命中 | Top1准确率 | Top3命中 | Top3准确率 |")
    md.append("|---|---|---|---|---|---|---|")
    for emo_en, v in sorted(whisper_res.per_emotion.items(), key=lambda x: -x[1]["n"]):
        md.append(f"| {EMOTION_CN.get(emo_en,emo_en)} | {emo_en} | {v['n']} | {v['top1']} | {v['top1_acc']*100:.0f}% | {v['top3']} | {v['top3_acc']*100:.0f}% |")

    # 3. LLM
    M = max(1, llm_res.total_samples)
    ts2 = llm_res.training_state or {}
    md.append(f"\n## 3. 🧠 LLM LoRA 评估 — `{Path(llm_res.model_path).relative_to(BASE_DIR)}`")
    md.append(f"\n### 3.1 训练状态")
    md.append(f"- Global step: **{ts2.get('global_step','-')}** / Epoch: **{ts2.get('epoch','-')}**")
    md.append(f"- Last train loss: `{ts2.get('last_train_loss','-')}`  LR: `{ts2.get('last_train_lr','-')}`")
    md.append(f"- Last eval loss: `{ts2.get('last_eval_loss','-')}`  (eval time {ts2.get('last_eval_runtime_s','-')}s)")

    md.append(f"\n### 3.2 核心指标 (N={llm_res.total_samples})")
    md.append(f"| 指标 | 值 | 说明 |")
    md.append("|---|---|---|")
    md.append(f"| 推理耗时 | `{llm_res.elapsed_s}s`  ({llm_res.elapsed_s/M:.2f}s/条) | Token 生成全流程 |")
    md.append(f"| 情绪分类 Top1 | **{llm_res.emotion_top1_hit}/{M} ({llm_res.emotion_top1_hit/M*100:.1f}%)** | |")
    md.append(f"| 情绪分类 Top3 | **{llm_res.emotion_top3_hit}/{M} ({llm_res.emotion_top3_hit/M*100:.1f}%)** | |")
    md.append(f"| 含义描述覆盖率 | **{llm_res.meaning_covered}/{M} ({llm_res.meaning_covered/M*100:.0f}%)** | 输出含 '含义/表达/情绪/需求' 等 |")
    md.append(f"| 行动建议覆盖率 | **{llm_res.advice_covered}/{M} ({llm_res.advice_covered/M*100:.0f}%)** | 输出含 '建议/应该/可以' 等 |")
    md.append(f"| 平均输出字数 | {llm_res.avg_output_len} 字 |  |")

    md.append(f"\n### 3.3 逐样本 LLM 明细 (预期 vs 真实 vs 命中)")
    md.append("| # | 宠物 | 场景/真实情绪 | 预测情绪Top3 | 含含义 | 含建议 | Top1命中 | Top3命中 | 输出预览 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in llm_res.samples:
        md.append(f"| {s['idx']} | {s['pet']} | {s['scene']}<br>`{s['emotion']}` | {s['pred_top3_emotions']} | {'✅' if s['has_meaning'] else '❌'} | {'✅' if s['has_advice'] else '❌'} | {'✅' if s['emo_top1_hit'] else '❌'} | {'✅' if s['emo_top3_hit'] else '❌'} | <pre>{s['output_preview']}</pre> |")

    md.append(f"\n### 3.4 分情绪 LLM 指标")
    md.append("| 情绪(中) | 情绪(en) | N | 情绪Top1Acc | 情绪Top3Acc | 含义覆盖 | 建议覆盖 |")
    md.append("|---|---|---|---|---|---|---|")
    for emo_en, v in sorted(llm_res.per_emotion.items(), key=lambda x: -x[1]["n"]):
        md.append(f"| {EMOTION_CN.get(emo_en,emo_en)} | {emo_en} | {v['n']} | {v['emo_top1_acc']*100:.0f}% | {v['emo_top3_acc']*100:.0f}% | {v['meaning_rate']*100:.0f}% | {v['advice_rate']*100:.0f}% |")

    # 4. E2E
    E = max(1, e2e_res.total_samples)
    md.append(f"\n## 4. 🔗 端到端 pet_to_text 评估 (Whisper→RAG→LLM)")
    md.append(f"\n### 4.1 核心指标 (N={e2e_res.total_samples})")
    pct = lambda a: f"{a/E*100:.1f}%" if E else "-"
    md.append(f"| 指标 | 命中数 | 准确率 |")
    md.append("|---|---|---|")
    md.append(f"| RAG 检索 Top1 情绪命中 | {e2e_res.rag_top1_hit}/{E} | **{pct(e2e_res.rag_top1_hit)}** |")
    md.append(f"| RAG 检索 Top3 情绪命中 | {e2e_res.rag_top3_hit}/{E} | **{pct(e2e_res.rag_top3_hit)}** |")
    md.append(f"| LLM 分析 Top1 情绪命中 | {e2e_res.llm_top1_hit}/{E} | **{pct(e2e_res.llm_top1_hit)}** |")
    md.append(f"| LLM 分析 Top3 情绪命中 | {e2e_res.llm_top3_hit}/{E} | **{pct(e2e_res.llm_top3_hit)}** |")
    md.append(f"| 端到端耗时 | {e2e_res.elapsed_s}s | {e2e_res.elapsed_s/E:.2f}s/条 |")

    md.append(f"\n### 4.2 逐样本 E2E 明细 (预期 vs 真实)")
    md.append("| # | 宠物 | 预期情绪 | Whisper(真实) | RAG Top3(真实) | RAG命中 | LLM Top3(真实) | LLM命中 | 输出预览 |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for s in e2e_res.samples:
        md.append(
            f"| {s['idx']} | {s['pet']} | {s['emotion']} | `{s['whisper_text']}` | {s['rag_top3']} | "
            f"{'✅T1' if s['rag_top1_hit'] else ('🔶T3' if s['rag_top3_hit'] else '❌')} | {s['llm_top3']} | "
            f"{'✅T1' if s['llm_top1_hit'] else ('🔶T3' if s['llm_top3_hit'] else '❌')} | {s['analysis_preview']} |"
        )

    md.append(f"\n### 4.3 分情绪 E2E 对比")
    md.append("| 情绪(中) | N | RAG Top1Acc | RAG Top3Acc | LLM Top1Acc | LLM Top3Acc |")
    md.append("|---|---|---|---|---|---|")
    for emo_en, v in sorted(e2e_res.per_emotion.items(), key=lambda x: -x[1]["n"]):
        md.append(f"| {EMOTION_CN.get(emo_en,emo_en)} | {v['n']} | {v['rag_t1_acc']*100:.0f}% | {v['rag_t3_acc']*100:.0f}% | {v['llm_t1_acc']*100:.0f}% | {v['llm_t3_acc']*100:.0f}% |")

    # 5. 改进建议
    md.append(f"\n## 5. ⚠️ 模型需要改进的点 (按优先级)")
    for i, it in enumerate(improvements, 1):
        md.append(f"\n### P{i} [{it['level']}] {it['model']}: {it['problem']}")
        md.append(f"- **根因**: {it['root']}")
        md.append("- **建议解决方案**:")
        for f in it["fix"]:
            md.append(f"  - {f}")
    return "\n".join(md)


def main():
    print("🔍 自动扫描 checkpoints/ 目录...")
    found = discover_models()
    print(f"   发现 LLM LoRA: {[m['name'] for m in found['llm']]}")
    print(f"   发现 Whisper LoRA: {[m['name'] for m in found['whisper']]}")
    if not found["llm"]:
        print("   ⚠️  未发现 LLM 模型, LLM/E2E 评估跳过")
    if not found["whisper"]:
        print("   ⚠️  未发现 Whisper 模型, Whisper/E2E 评估跳过")

    gt = build_ground_truth()
    print(f"\n📚 评估数据集: {len(gt)} 条 (猫 + 狗, 覆盖 {len(set(g['emotion_en'] for g in gt))} 种情绪)")

    # 取最新的 checkpoint
    llm_m     = found["llm"][0]     if found["llm"]     else None
    whisper_m = found["whisper"][0] if found["whisper"] else None

    w_res = evaluate_whisper(whisper_m, gt) if whisper_m else WhisperEvalResult(model_path="N/A", base_model="N/A")
    l_res = evaluate_llm(llm_m, gt)           if llm_m     else LLMEvalResult(model_path="N/A", base_model="N/A")
    e_res = evaluate_e2e(w_res, l_res, gt, whisper_m, llm_m) if (whisper_m and llm_m) else E2EEvalResult()

    # 改进建议
    improvements = build_improvements(w_res, l_res, e_res)

    # 写 JSON + MD
    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "discovered_models": found,
        "whisper_eval": asdict(w_res) if isinstance(w_res, WhisperEvalResult) else w_res,
        "llm_eval":     asdict(l_res) if isinstance(l_res, LLMEvalResult) else l_res,
        "e2e_eval":     asdict(e_res) if isinstance(e_res, E2EEvalResult) else e_res,
        "improvements": improvements,
    }
    REPORT_JSON.write_text(json.dumps(report_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 JSON 报告已保存: {REPORT_JSON}")

    md_text = build_markdown_report(found, w_res, l_res, e_res, improvements)
    REPORT_MD.write_text(md_text, encoding="utf-8")
    print(f"💾 Markdown 报告已保存: {REPORT_MD}")

    # 打印最终摘要
    print("\n" + "="*80)
    print("📊 模型评估最终摘要")
    print("="*80)
    N = max(1, w_res.total_samples)
    M = max(1, l_res.total_samples)
    E = max(1, e_res.total_samples)
    print(f"🎙️  Whisper   Top1={w_res.top1_hit/N*100:5.1f}%  Top3={w_res.top3_hit/N*100:5.1f}%  "
          f"Fallback={w_res.fallback_outputs/N*100:4.0f}%  Empty={w_res.empty_outputs}")
    if M:
        print(f"🧠  LLM       Top1={l_res.emotion_top1_hit/M*100:5.1f}%  Top3={l_res.emotion_top3_hit/M*100:5.1f}%  "
              f"含义={l_res.meaning_covered/M*100:4.0f}%  建议={l_res.advice_covered/M*100:4.0f}%")
    if E:
        print(f"🔗  RAG       Top1={e_res.rag_top1_hit/E*100:5.1f}%  Top3={e_res.rag_top3_hit/E*100:5.1f}%")
        print(f"🔗  LLM(E2E)  Top1={e_res.llm_top1_hit/E*100:5.1f}%  Top3={e_res.llm_top3_hit/E*100:5.1f}%")
    print(f"\n⚠️  P0/P1 改进建议数: {len([i for i in improvements if i['level'] in ('P0','P1')])}")
    for it in improvements[:4]:
        print(f"   - [{it['level']}][{it['model']}] {it['problem'][:60]}")

if __name__ == "__main__":
    main()
