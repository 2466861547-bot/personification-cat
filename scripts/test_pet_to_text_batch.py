#!/usr/bin/env python3
"""批量测试 pet_to_text 模式: cat + dog 各音频
收集完整流程数据 (Whisper ASR / RAG 检索 / LLM 分析) 并输出评估报告
"""
import os
import sys
import json
import time
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from src.inference.pipeline import TranslationPipeline

CAT_TESTS = [
    ("hungry_feeding_time.wav", "橘猫", "hungry"),
    ("happy_purring.wav",      "英短", "happy"),
    ("angry_hissing.wav",      "暹罗", "angry"),
    ("fear_shaky.wav",         "缅因", "fear"),
    ("seek_attention_petme.wav", "加菲猫", "seek_attention"),
    ("pain_injured.wav",       "苏格兰折耳", "pain"),
    ("alert_bird_watching.wav","孟加拉豹猫", "alert"),
    ("excited_treat_time.wav", "矮脚拿破仑", "excited"),
    ("anxious_vet_visit.wav",  "斯芬克斯无毛猫", "anxious"),
    ("content_blanket.wav",    "布偶", "content"),
    ("frustrated_door_closed.wav","狸花", "frustrated"),
    ("lonely_home_alone.wav",  "波斯", "lonely"),
]

DOG_TESTS = [
    ("hungry_dinner_time.wav", "柴犬",   "hungry"),
    ("happy_owner_home.wav",   "金毛",   "happy"),
    ("angry_stranger.wav",     "罗威纳", "angry"),
    ("fear_thunder.wav",       "比熊",   "fear"),
    ("seek_attention_walk.wav","边牧",   "seek_attention"),
    ("pain_stepped_on.wav",    "泰迪",   "pain"),
    ("alert_doorbell.wav",     "德牧",   "alert"),
    ("excited_leash_time.wav", "哈士奇", "excited"),
    ("territorial_fence.wav",  "藏獒",   "territorial"),
    ("greeting_welcome.wav",   "拉布拉多","greeting"),
    ("anxious_alone.wav",      "雪纳瑞", "anxious"),
    ("playful_fetch.wav",      "柯基",   "playful"),
]

def run_batch(pet_type, tests):
    print(f"\n\n{'#'*72}")
    print(f"# 批量测试 pet_to_text: {pet_type.upper()} ({len(tests)} 个样本)")
    print(f"{'#'*72}")

    # 一次性初始化组件 (复用 Whisper/LLM/RAG)
    print(f"\n⏳ 初始化 pipeline (加载 Whisper/LLM/RAG)...")
    t0 = time.time()
    pipe = TranslationPipeline(
        whisper_model_path=None,
        llm_adapter_path=os.path.join(BASE_DIR, "checkpoints/llm_mps/checkpoint-300"),
        use_rag=True,
        use_agent=False,
    )
    t_init = time.time() - t0
    print(f"⏱️  初始化耗时: {t_init:.1f}s")

    results = []
    whisper_texts = defaultdict(list)     # emotion -> [text, text, ...]
    rag_hit_emotions = defaultdict(list)  # true_emotion -> [Top1 emo, Top2 emo, ...]

    for filename, breed, true_emo in tests:
        if pet_type == "cat":
            audio_path = os.path.join(BASE_DIR, "data/raw/cat_sounds", filename)
        else:
            audio_path = os.path.join(BASE_DIR, "data/raw/dog_sounds", filename)

        if not os.path.exists(audio_path):
            print(f"\n⚠️  文件不存在, 跳过: {filename}")
            continue

        print(f"\n{'─'*72}")
        print(f"▶ [{pet_type.upper()}] {filename}  (真实情绪: {true_emo}, 品种: {breed})")
        print(f"  文件大小: {os.path.getsize(audio_path)/1024:.1f}KB")

        t = time.time()
        # 调用 pipeline (内部已打印所有业务架构分层日志)
        # 为避免日志过大, 重定向 stdout 捕获
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = pipe.pet_sound_to_text(audio_path, pet_type=pet_type, breed=breed)
        elapsed = time.time() - t

        whisper_text = res.get("whisper_transcription", "")
        rag_hits = res.get("rag_details", [])
        analysis = res.get("analysis", "")

        # Whisper 文本
        whisper_texts[true_emo].append(whisper_text)

        # RAG Top3 情绪
        rag_top_emotions = []
        for r in rag_hits[:3]:
            emo = r.get("metadata", {}).get("emotion", "")
            if emo:
                rag_top_emotions.append(emo)
        rag_hit_emotions[true_emo] = rag_top_emotions

        # Top1 是否命中
        rag_top1_hit = (rag_top_emotions[0] == true_emo) if rag_top_emotions else False
        rag_top3_hit = (true_emo in rag_top_emotions) if rag_top_emotions else False

        print(f"  ⏱️  推理耗时: {elapsed:.2f}s")
        print(f"  📝 Whisper 输出: '{whisper_text}'")
        print(f"  🧠 RAG 命中情绪 (Top3): {rag_top_emotions}")
        print(f"  🎯 RAG Top1 命中: {'✅' if rag_top1_hit else '❌'}   Top3 命中: {'✅' if rag_top3_hit else '❌'}")
        if len(analysis) > 150:
            print(f"  📄 LLM 分析 (前150字): {analysis[:150]}...")
        else:
            print(f"  📄 LLM 分析: {analysis}")

        results.append({
            "pet": pet_type,
            "filename": filename,
            "breed": breed,
            "true_emotion": true_emo,
            "whisper_text": whisper_text,
            "rag_top3": rag_top_emotions,
            "rag_top1_hit": rag_top1_hit,
            "rag_top3_hit": rag_top3_hit,
            "analysis": analysis,
            "elapsed_s": round(elapsed, 2),
        })

    return results


def print_report(results, pet_type):
    print(f"\n\n{'='*72}")
    print(f"📊 PET_TO_TEXT 批量测试报告 — {pet_type.upper()} (N={len(results)})")
    print(f"{'='*72}")

    # 1. Whisper 识别多样性
    all_whisper = Counter(r["whisper_text"] for r in results)
    print(f"\n1️⃣  Whisper ASR 识别结果分布:")
    print(f"     不同识别结果数: {len(all_whisper)} / {len(results)}")
    for text, cnt in all_whisper.most_common(5):
        pct = cnt/len(results)*100
        print(f"       '{text}': {cnt} 次 ({pct:.0f}%)")

    # 2. RAG 命中率
    top1_correct = sum(1 for r in results if r["rag_top1_hit"])
    top3_correct = sum(1 for r in results if r["rag_top3_hit"])
    print(f"\n2️⃣  RAG 情绪检索准确率:")
    print(f"     Top1 准确率: {top1_correct}/{len(results)} = {top1_correct/len(results)*100:.1f}%")
    print(f"     Top3 准确率: {top3_correct}/{len(results)} = {top3_correct/len(results)*100:.1f}%")

    # 3. 每情绪详细
    print(f"\n3️⃣  逐样本详情:")
    print(f"   {'真实情绪':14s} {'Whisper':18s} {'RAG Top1':12s} {'Top3命中':10s} {'备注'}")
    print(f"   {'─'*90}")
    for r in results:
        hit = "✅" if r["rag_top1_hit"] else ("🔶" if r["rag_top3_hit"] else "❌")
        rag = r["rag_top3"][0] if r["rag_top3"] else "-"
        remark = f"Top2/3={r['rag_top3'][1:] if len(r['rag_top3'])>1 else '-'}" if not r["rag_top1_hit"] and r["rag_top3_hit"] else ""
        if not r["rag_top3_hit"]:
            remark = f"RAG 完全未命中"
        print(f"   {r['true_emotion']:14s} {r['whisper_text']:18s} {rag:12s} {hit:10s} {remark}")

    # 4. 流程欠缺分析
    print(f"\n4️⃣  ⚠️  pet_to_text 流程欠缺分析:")
    issue_counter = Counter()

    # 4a) Whisper 全是 fallback 单字 (猫=嗚/狗=汪汪类)
    fallback_words = {"嗚", "呜", "汪", "汪汪", "吠", "", "叫"}
    whisper_fallback_count = sum(1 for r in results if r["whisper_text"].strip() in fallback_words or len(r["whisper_text"].strip()) <= 2)
    if whisper_fallback_count:
        issue_counter["Whisper 识别退化(≤2字乱码)"] = whisper_fallback_count

    # 4b) RAG Top1 未命中
    issue_counter["RAG Top1 未命中"] = len(results) - top1_correct
    issue_counter["RAG Top3 未命中"] = len(results) - top3_correct

    # 4c) Whisper 为空
    empty_whisper = sum(1 for r in results if not r["whisper_text"].strip())
    if empty_whisper:
        issue_counter["Whisper 输出为空"] = empty_whisper

    for issue, cnt in issue_counter.most_common():
        pct = cnt/len(results)*100
        print(f"     ❌ {issue}: {cnt}/{len(results)} ({pct:.0f}%)")

    return results


def main():
    cat_results = run_batch("cat", CAT_TESTS)
    cat_report = print_report(cat_results, "cat")

    dog_results = run_batch("dog", DOG_TESTS)
    dog_report = print_report(dog_results, "dog")

    # 汇总保存
    all_data = {
        "cat": cat_results,
        "dog": dog_results,
    }
    out_path = os.path.join(BASE_DIR, "output", "pet_to_text_batch_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 详细报告已保存: {out_path}")

if __name__ == "__main__":
    main()
