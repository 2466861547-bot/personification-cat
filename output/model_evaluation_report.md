# 🧪 模型评估报告 — 宠物声音翻译系统

> 评估时间: 2026-08-24 14:33:59  
> 自动扫描 `checkpoints/` 发现模型: **LLM=3 个, Whisper=2 个**

## 1. 已发现训练模型

### LLM LoRA
| # | 名称 | 路径 | 基座模型 |
|---|---|---|---|
| 1 | llm_mps/checkpoint-300 | `checkpoints/llm_mps/checkpoint-300` | `tinyllama/tinyllama-1.1b-chat-v1.0` |
| 2 | llm_mps/checkpoint-250 | `checkpoints/llm_mps/checkpoint-250` | `tinyllama/tinyllama-1.1b-chat-v1.0` |
| 3 | llm_mps/checkpoint-200 | `checkpoints/llm_mps/checkpoint-200` | `tinyllama/tinyllama-1.1b-chat-v1.0` |

### Whisper LoRA
| # | 名称 | 路径 | 基座模型 |
|---|---|---|---|
| 1 | final | `checkpoints/whisper/final` | `openai/whisper-tiny` |
| 2 | checkpoint-315 | `checkpoints/whisper/checkpoint-315` | `openai/whisper-tiny` |

## 2. 🎙️ Whisper ASR 评估 — `checkpoints/whisper/final`

### 2.1 训练状态
- Global step: **-** / Epoch: **-**
- Last train loss: `-`  LR: `-`
- Last eval loss: `-`  (eval time -s)
- Best checkpoint: `-`

### 2.2 核心指标 (N=22)
| 指标 | 值 | 说明 |
|---|---|---|
| 推理耗时 | `14.27s`  (0.65s/条) | 含 Mel+generate 全流程 |
| Top1 情绪关键词命中 | **2/22 (9.1%)** | 关键词法第1情绪 = 真实情绪 |
| Top3 情绪关键词命中 | **6/22 (27.3%)** | 前3候选包含真实情绪 |
| 独立输出文本数 | 16/22 | 衡量识别多样性 (越低越退化) |
| 空输出 | 0 (0%) | 完全没有输出 |
| Fallback 输出 (≤2字) | **3 (14%)** | 嗚/汪/呜 等退化结果 |

### 2.3 Whisper 输出 Top 分布
```
  '啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊' × 3  (14%)
  '你' × 3  (14%)
  '啊' × 2  (9%)
  '嗚嗚嗚嗚' × 2  (9%)
  '咱们快快快' × 1  (5%)
  '小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伴伴们的小伴伴伴们的小伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴�' × 1  (5%)
  '走走走' × 1  (5%)
  'FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU' × 1  (5%)
```

### 2.4 Whisper 逐样本明细 (预期 vs 真实 vs 命中)
| # | 宠物 | 场景/真实情绪 | 🎯 Whisper 真实输出 | 预期情绪Top3 | Top1命中 | Top3命中 |
|---|---|---|---|---|---|---|
| 1 | cat | alert bird watching<br>`alert(警惕)` | `咱们快快快` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 2 | cat | angry hissing<br>`angry(愤怒)` | `啊` | ['hungry', 'happy', 'angry'] | ❌ | ✅ |
| 3 | cat | anxious vet visit<br>`anxious(焦虑)` | `嗚嗚嗚嗚` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 4 | cat | content blanket<br>`content(满足)` | `小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伴伴们的小伴伴伴们的小伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴�` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 5 | cat | excited treat time<br>`excited(兴奋)` | `走走走` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 6 | cat | fear shaky<br>`fear(恐惧)` | `FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 7 | cat | frustrated door closed<br>`frustrated(挫败)` | `嗯` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 8 | cat | happy purring<br>`happy(开心)` | `嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚` | ['hungry', 'happy', 'angry'] | ❌ | ✅ |
| 9 | cat | hungry feeding time<br>`hungry(饥饿)` | `嗚嗚嗚嗚` | ['hungry', 'happy', 'angry'] | ✅ | ✅ |
| 10 | cat | lonely home alone<br>`lonely(孤独)` | `嗚嗚嗚嗚嗚` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 11 | cat | pain injured<br>`pain(疼痛)` | `我` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 12 | dog | alert doorbell<br>`alert(警惕)` | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 13 | dog | angry stranger<br>`angry(愤怒)` | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['hungry', 'happy', 'angry'] | ❌ | ✅ |
| 14 | dog | anxious alone<br>`anxious(焦虑)` | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 15 | dog | excited leash time<br>`excited(兴奋)` | `你` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 16 | dog | fear thunder<br>`fear(恐惧)` | `好` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 17 | dog | greeting welcome<br>`greeting(问候)` | `哈哈哈哈` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 18 | dog | happy owner home<br>`happy(开心)` | `你` | ['hungry', 'happy', 'angry'] | ❌ | ✅ |
| 19 | dog | hungry dinner time<br>`hungry(饥饿)` | `啊` | ['hungry', 'happy', 'angry'] | ✅ | ✅ |
| 20 | dog | pain stepped on<br>`pain(疼痛)` | `嘗嘗嘗` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 21 | dog | playful fetch<br>`playful(玩耍)` | `你` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |
| 22 | dog | territorial fence<br>`territorial(领地意识)` | `哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦` | ['hungry', 'happy', 'angry'] | ❌ | ❌ |

### 2.5 分情绪 Whisper 准确率
| 情绪(中) | 情绪(en) | 样本数 | Top1命中 | Top1准确率 | Top3命中 | Top3准确率 |
|---|---|---|---|---|---|---|
| 警惕 | alert | 2 | 0 | 0% | 0 | 0% |
| 愤怒 | angry | 2 | 0 | 0% | 2 | 100% |
| 焦虑 | anxious | 2 | 0 | 0% | 0 | 0% |
| 兴奋 | excited | 2 | 0 | 0% | 0 | 0% |
| 恐惧 | fear | 2 | 0 | 0% | 0 | 0% |
| 开心 | happy | 2 | 0 | 0% | 2 | 100% |
| 饥饿 | hungry | 2 | 2 | 100% | 2 | 100% |
| 疼痛 | pain | 2 | 0 | 0% | 0 | 0% |
| 满足 | content | 1 | 0 | 0% | 0 | 0% |
| 挫败 | frustrated | 1 | 0 | 0% | 0 | 0% |
| 孤独 | lonely | 1 | 0 | 0% | 0 | 0% |
| 问候 | greeting | 1 | 0 | 0% | 0 | 0% |
| 玩耍 | playful | 1 | 0 | 0% | 0 | 0% |
| 领地意识 | territorial | 1 | 0 | 0% | 0 | 0% |

## 3. 🧠 LLM LoRA 评估 — `checkpoints/llm_mps/checkpoint-300`

### 3.1 训练状态
- Global step: **300** / Epoch: **1.3333333333333333**
- Last train loss: `0.011092793941497803`  LR: `0.00011498470948012233`
- Last eval loss: `0.009457923471927643`  (eval time 3914.4427s)

### 3.2 核心指标 (N=22)
| 指标 | 值 | 说明 |
|---|---|---|
| 推理耗时 | `498.54s`  (22.66s/条) | Token 生成全流程 |
| 情绪分类 Top1 | **19/22 (86.4%)** | |
| 情绪分类 Top3 | **20/22 (90.9%)** | |
| 含义描述覆盖率 | **19/22 (86%)** | 输出含 '含义/表达/情绪/需求' 等 |
| 行动建议覆盖率 | **18/22 (82%)** | 输出含 '建议/应该/可以' 等 |
| 平均输出字数 | 226.8 字 |  |

### 3.3 逐样本 LLM 明细 (预期 vs 真实 vs 命中)
| # | 宠物 | 场景/真实情绪 | 预测情绪Top3 | 含含义 | 含建议 | Top1命中 | Top3命中 | 输出预览 |
|---|---|---|---|---|---|---|---|---|
| 1 | cat | alert bird watching<br>`alert(警惕)` | ['alert'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: cat
声音描述: 发出警惕的叫声，场景: alert bird watching
知识库参考:
  - [alert] 声音: 典型警惕声音
  - 含 jú警惕叫声音描述: 宠物正在表明警惕的情绪
  - 建议: 请根 Außer警惕叫声音描述的建议措施

请出现此类声音属于宠物警惕的表现，需要关注于安全措施措施。</pre> |
| 2 | cat | angry hissing<br>`angry(愤怒)` | ['angry'] | ❌ | ❌ | ✅ | ✅ | <pre>宠物类型: cat
声音描述: 发出愤怒的叫声，场景: angry hissing
知识库参考:
  - [ANGRY]声音:宠物正常的叫声音况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况声音类况</pre> |
| 3 | cat | anxious vet visit<br>`anxious(焦虑)` | ['anxious', 'hungry'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物焦虑的状态。

##声音描述读读分析
声音描述读读分析： 发出焦虑的叫声，场景: anxious vetVISION
这种叫声通常表明宠物焦虑的状态。

##建议措施
请根 Transfermarkt于情绪解读，检查上次喂食时间，适量补充食物，宠物状态安静后继续互动。
</pre> |
| 4 | cat | content blanket<br>`content(满足)` | ['content', 'happy'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物满足的状态。

##声音描述读读解读
声音描述读读解读： 发出满足的叫声，场景: contentblanket
这种叫声通常表明宠物满足的状态。

##建议措施读读解读
请根 Wikiped读读解读： 叫声描述读读解读： 发出满足的叫声，场景: contentblanket
这种叫声通常表明宠物满足的状态。

##建议措施读读解读
请根读读解读： 叫声描述读读解读</pre> |
| 5 | cat | excited treat time<br>`excited(兴奋)` | ['excited'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物情绪正常的状态。

##声音描述读读分析
声音描述读读分析： 发出兴奋的叫声，场景: externsuitetime
这种叫声通常表明宠物情绪正常的状态。

##建议措施
请根 фамилией做出适当反应。

##情绪解读读读分析
声音描述读读分析： 分析叫声通常表明宠物情绪正常的状态。
这种叫声通常表明宠物情绪正常的状态。

##建议措施
请根��������</pre> |
| 6 | cat | fear shaky<br>`fear(恐惧)` | ['fear'] | ✅ | ❌ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物恐惧的状态。

##声音描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读读情绪解读描述读</pre> |
| 7 | cat | frustrated door closed<br>`frustrated(挫败)` | ['hungry', 'frustrated'] | ❌ | ✅ | ❌ | ✅ | <pre>这种叫声通常表明宠物挫euw的状态。
声音描述: 发出挫euw的叫声，场景: frustrated door closed
知识库参考:
  - [fr�����������断叫声分析况况措施](https://www.sciencedirectarticlesbrasilia.com/article/piiratbn00020003)
诊断方向措施：请根 jú施诊断声音分析况况措施
建议措施：请</pre> |
| 8 | cat | happy purring<br>`happy(开心)` | ['happy'] | ✅ | ✅ | ✅ | ✅ | <pre>这种叫声通常表明宠物开心的情绪。
声音描述: 发出开心的叫声，场景: happy prufkowitz
知识库参考:
  - [开心声音叫声解读](https://zhiaiyouzi.github.«建议: 请根 Siehe解读叫声解读的方向，可以检查其情绪状态。)
请输出以下格式:
1. 情绪解读: <情绪读读>
2. 具体叫声分析：<叫声分析叫声解读的方向，可以检查其情绪状态。)
建议检查其</pre> |
| 9 | cat | hungry feeding time<br>`hungry(饥饿)` | ['hungry'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物饥饿的状态。

##声音描述读读检查
声音描述读读检查出现此类声音属于需要关注的表现。

##建议措施
请根 Siehe饥饿声音的建议措施，适量补充食物，宠物状态 automatisch适量补充食物，保持环境安静，不要强迫互动，适量补充食物，宠物状态 automatisch适量补充食物，保持环境安静，不要强追补充食物，适量补充食物，宠物状态 automatisch</pre> |
| 10 | cat | lonely home alone<br>`lonely(孤独)` | ['lonely', 'hungry', 'seek_attention'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: cat
声音描述: 发出孤独的叫声，场景: lonely home alone
知识库参考:
  - [lonely] 声音: 典Ť特况孤独声音
  - 含义: 宠物正在表明孤独的情绪。请检查上次喂食时间，适量补充食物，适量互动抚摸，继续维持良好的互动关系。
请出现此类声音属于叫声通常表明宠物孤独的情绪。建议检查上次喂食时间，适量补充食物，适量互动抚摸，继续维持良好的互动关系。</pre> |
| 11 | cat | pain injured<br>`pain(疼痛)` | ['pain', 'seek_attention'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: cat
品种: 通用
声音描述: 发出疼痛的叫声，场景: pain injured
知识库参考:
  - [疼痛声音」 声音描述: 宠物正在表明疼痛的情绪」
  - 含义: 宠物正在表明疼痛的情绪」
  - 建议: 请根 Wikiped议情绪解读，适度抚摸疼痛的情绪解读，保持环境安静，不要强迫互动，等其平静后再接近。

请出现此类叫声音描述：宠物正在表明疼痛的情绪」。建议抚摸疼痛的情绪</pre> |
| 12 | dog | alert doorbell<br>`alert(警惕)` | ['alert', 'hungry'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: dog
品种: 通用
声音描述: 发出警惕的叫声，场景: alert doorbell
知识库参考:
  - [alert] 声音: 典 techniques: 宠物正常的声音，表明正常的状态。
  - 含义: 宠物正常的情绪解读。
  - 建议: 请根Ź�����断，确认安全后安抚其情绪。

请出现此类声音属于需要关注的表现，建议检查上次喂食时间，适量补充食物，继续维持良好的互动关系</pre> |
| 13 | dog | angry stranger<br>`angry(愤怒)` | ['angry', 'alert'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: dog
品种: 通用
声音描述: 发出愤怒的叫声，场景: angry stranger
知识库参考:
  - [angrar愤怒声音] 声音: 典型愤怒声音
  - 含义: 宠物正在表明愤怒的情绪。请检查身体状况，联系兽医进行专业诊断。
兽医建议检查身体状况，联系兽医进行专业诊断。</pre> |
| 14 | dog | anxious alone<br>`anxious(焦虑)` | ['anxious', 'lonely'] | ❌ | ❌ | ✅ | ✅ | <pre>宠物类型: dog
品种: 通用
声音描述: 发出焦虑的叫声，场景: anxiousalone
知识库参考:
  - [焦虑声音读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查叫声分析读音措施检查</pre> |
| 15 | dog | excited leash time<br>`excited(兴奋)` | ['hungry', 'happy', 'angry'] | ✅ | ❌ | ❌ | ❌ | <pre>##情绪解读
这种叫声通常表明宠物情绪解读正常的状态。

##声音描述读读分析
声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分析：声音描述读读分</pre> |
| 16 | dog | fear thunder<br>`fear(恐惧)` | ['fear'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: dog
声音描述: 发出恐惧的叫声，场景: fear thunder
知识库参考:
  - [f weap�������] 声音: 典型恐惧声音
  - 含义: 宠物正在表明恐惧的情绪。请检查安全诊断方向。
  - 建议: 请根Ź建议: 请检查安全诊断方向。

请出现此类声音属于需要关注的表现。请检查安全诊断方向。</pre> |
| 17 | dog | greeting welcome<br>`greeting(问候)` | ['hungry', 'happy', 'angry'] | ✅ | ✅ | ❌ | ❌ | <pre>##情绪解读
这种叫声通常表明宠物正常的状态。

##声音描述读读解读
声音描述读读解读：叫声通常表明宠物正常的状态。

##建议措施读读解读
请根 hina读读解读：叫声通常表明宠物正常的状态。

##情绪解读读读解读
情绪解读读读解读：声音描述读读解读：叫声通常表明宠物正常的状态。

##建议措施读读解读
请根 hina读读解读：叫声通常表明宠物正常的状态。

##情绪解读读读解读
情绪解读读</pre> |
| 18 | dog | happy owner home<br>`happy(开心)` | ['happy', 'lonely'] | ✅ | ✅ | ✅ | ✅ | <pre>这种叫声通常表明宠物开心的情绪。
声音描述: 发出开心的叫声，场景: happyOwnerhome
知识库参考:
  - [开心声音] 声音: 典型开心声音
  - 含义: 宠物正在表明开心的情绪
  - 建议: 请根 Wikipedias上的建议措施
声音描述: 发出开心的叫声，场景: happyOwnerhome
建议措施: 请根 Wikipedias上的建议措施
</pre> |
| 19 | dog | hungry dinner time<br>`hungry(饥饿)` | ['hungry'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物饥饿的状态。

##声音描述读读叫声通常表明宠物饥饿的状态。
声音描述:  daughteck饥饿声音
叫声通常表明宠物饥饿的状态。

##建议措施读读叫声通常表明宠物饥饿的状态。
请根 Wikipedias上的叫声分析，确认声音描述读读叫声通常表明宠物饥饿的状态。

##建议措施读读叫声通常表明宠物饥饿的状态。
请根读读声音描述读读叫声通常表明宠物饥饿的状态。
</pre> |
| 20 | dog | pain stepped on<br>`pain(疼痛)` | ['pain'] | ✅ | ✅ | ✅ | ✅ | <pre>宠物类型: dog
品种: 通用
声音描述: 发出疼痛的叫声，场景: pain stepped on
知识库参考:
  - [疼痛声音读读解读](https://zhijd疼痛声音读读解读)
读读解读分析宠物情绪解读的方向。声音描述：宠物正在表明疼痛的情绪。建议措施：请根 Биография充塞施足够的互动，避免直视，等其平静后再接近。
</pre> |
| 21 | dog | playful fetch<br>`playful(玩耍)` | ['playful', 'seek_attention'] | ✅ | ✅ | ✅ | ✅ | <pre>##声音描述：宠物正在表明玩耍的情绪。
声音描述: 发出玩耍的叫声，场景: playful fetch
知识库参考:
  - [playsuit]声音：玩耍的叫声音读明情绪。
读明情绪做出适 WorldCat建议:请根 Transfermarkt读明情绪。
建议措施：请根明情绪做出适当建议。
</pre> |
| 22 | dog | territorial fence<br>`territorial(领地意识)` | ['territorial', 'seek_attention'] | ✅ | ✅ | ✅ | ✅ | <pre>##情绪解读
这种叫声通常表明宠物正常的状态。

##声音描述读读分析
声音描述读读分析： 发出领地意识的叫声，场景: territorial fence
这种叫声通常表明宠物正常的状态。

##建议措施
请根 Wikiped议情绪解读的方向，适度互动抚摸，维持良好的互动关系。
</pre> |

### 3.4 分情绪 LLM 指标
| 情绪(中) | 情绪(en) | N | 情绪Top1Acc | 情绪Top3Acc | 含义覆盖 | 建议覆盖 |
|---|---|---|---|---|---|---|
| 警惕 | alert | 2 | 100% | 100% | 100% | 100% |
| 愤怒 | angry | 2 | 100% | 100% | 50% | 50% |
| 焦虑 | anxious | 2 | 100% | 100% | 50% | 50% |
| 兴奋 | excited | 2 | 50% | 50% | 100% | 50% |
| 恐惧 | fear | 2 | 100% | 100% | 100% | 50% |
| 开心 | happy | 2 | 100% | 100% | 100% | 100% |
| 饥饿 | hungry | 2 | 100% | 100% | 100% | 100% |
| 疼痛 | pain | 2 | 100% | 100% | 100% | 100% |
| 满足 | content | 1 | 100% | 100% | 100% | 100% |
| 挫败 | frustrated | 1 | 0% | 100% | 0% | 100% |
| 孤独 | lonely | 1 | 100% | 100% | 100% | 100% |
| 问候 | greeting | 1 | 0% | 0% | 100% | 100% |
| 玩耍 | playful | 1 | 100% | 100% | 100% | 100% |
| 领地意识 | territorial | 1 | 100% | 100% | 100% | 100% |

## 4. 🔗 端到端 pet_to_text 评估 (Whisper→RAG→LLM)

### 4.1 核心指标 (N=22)
| 指标 | 命中数 | 准确率 |
|---|---|---|
| RAG 检索 Top1 情绪命中 | 1/22 | **4.5%** |
| RAG 检索 Top3 情绪命中 | 3/22 | **13.6%** |
| LLM 分析 Top1 情绪命中 | 1/22 | **4.5%** |
| LLM 分析 Top3 情绪命中 | 3/22 | **13.6%** |
| 端到端耗时 | 1124.95s | 51.13s/条 |

### 4.2 逐样本 E2E 明细 (预期 vs 真实)
| # | 宠物 | 预期情绪 | Whisper(真实) | RAG Top3(真实) | RAG命中 | LLM Top3(真实) | LLM命中 | 输出预览 |
|---|---|---|---|---|---|---|---|---|
| 1 | cat | alert(警惕) | `咱们快快快` | ['excited', 'greeting', 'happy'] | ❌ | ['happy', 'hungry', 'content'] | ❌ | 这种宠物类型的叫声通常表明宠物心情愉悦的状态。在当前情况下，建议满足猫咪的期往喂食互动，适度互动或喂食，避免过idens |
| 2 | cat | angry(愤怒) | `啊` | ['frustrated', 'happy', 'excited'] | ❌ | ['happy', 'hungry', 'seek_attention'] | ❌ | 这种叫声通常表明宠物心情愉悦，可适度抚摸猫咪头部和下巴。声音描述: 啊 叫声通常表明宠物心情愉悦，可适度抚摸猫咪头部和下 |
| 3 | cat | anxious(焦虑) | `嗚嗚嗚嗚` | ['frustrated', 'happy', 'relaxed'] | ❌ | ['hungry', 'happy', 'angry'] | ❌ | 这种宠物类型的声音描述：嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚 |
| 4 | cat | content(满足) | `小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伙伴们的小伴伴们的小伴伴伴们的小伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴伴�` | ['happy', 'jealous', 'excited'] | ❌ | ['happy', 'seek_attention'] | ❌ | 这种宠物类型的声音描述： 轻柔的短喵叫，伴随呼噜声，音调平稳 这种宠物类型的情绪： happy 这种宠物类型的声音描述： |
| 5 | cat | excited(兴奋) | `走走走` | ['greeting', 'jealous', 'excited'] | 🔶T3 | ['seek_attention', 'playful', 'jealous'] | ❌ | 这种宠物类型的声音描述： 走走走 这种宠物类型情绪： seek_attention 声音描述： 持续且有节奏的喵叫，可e |
| 6 | cat | fear(恐惧) | `FUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU` | ['frustrated', 'anxious', 'angry'] | ❌ | ['hungry'] | ❌ | 这种叫声通常表明宠物情绪的状态。叫声特况读音描述：宠物猫咪感到饥饿，请求食物。情境: 通常在饭点前后，或看到食物相关物品 |
| 7 | cat | frustrated(挫败) | `嗯` | ['frustrated', 'happy', 'excited'] | ✅T1 | ['happy', 'hungry'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在舒适环境下，猫咪出现此类声音属于正常的表现。建议检查上次喂食时间，若超观察时间况况 |
| 8 | cat | happy(开心) | `嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚` | ['relaxed', 'confused', 'anxious'] | ❌ | ['hungry', 'happy', 'angry'] | 🔶T3 | 这种宠物类型的声音描述： ����������������������������������������������� |
| 9 | cat | hungry(饥饿) | `嗚嗚嗚嗚` | ['frustrated', 'happy', 'relaxed'] | ❌ | ['hungry', 'happy', 'angry'] | ✅T1 | 这种宠物类型的声音描述：嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚嗚 |
| 10 | cat | lonely(孤独) | `嗚嗚嗚嗚嗚` | ['anxious', 'happy', 'frustrated'] | ❌ | ['hungry', 'happy'] | ❌ | 这种宠物类型：cat，品种：通常不太爱叫，情绪：极轻柔的呼噜声，含蓄。建议: 英短易胖，注意控制食量，建议定时定量喂食  |
| 11 | cat | pain(疼痛) | `我` | ['frustrated', 'anxious', 'jealous'] | ❌ | ['seek_attention'] | ❌ | 这种宠物类型的声音描述： 警觉的猫咪感读叫声通常表明宠物状态正常的状态。 声音描述: 不规裂的猫咪感读叫声通常表明宠物状 |
| 12 | dog | alert(警惕) | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['jealous', 'happy', 'happy'] | ❌ | ['happy', 'seek_attention'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在宠物类型：叫声分析下读读情绪：「开心」，建议抚摸叫声分析下读读情绪：「开心」，建议 |
| 13 | dog | angry(愤怒) | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['jealous', 'happy', 'happy'] | ❌ | ['happy'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在宠物类型：叫声分析况况下，叫声通常表明宠物心情愉悦的状态。在宠物类型：萨摩耶分析况 |
| 14 | dog | anxious(焦虑) | `啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊` | ['jealous', 'happy', 'happy'] | ❌ | ['happy', 'seek_attention'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在宠物类型的情况下，建议抚摸叫声的环境安静，不要突然互动，避免强迫互动，保持环境安静 |
| 15 | dog | excited(兴奋) | `你` | ['anxious', 'jealous', 'happy'] | ❌ | ['alert'] | ❌ | 这种宠物类型的声音描述： 轻快短促的汪叫声音描述: 狗狗警觉到异常情况，发出警报。含 Wikipediai读读叫声音描述 |
| 16 | dog | fear(恐惧) | `好` | ['happy', 'happy', 'excited'] | ❌ | ['happy'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。叫声通常表明宠物心情愉悦的状态。在家轻快的情况下，建议检查叫声误音描述，确认安全后安 |
| 17 | dog | greeting(问候) | `哈哈哈哈` | ['happy', 'happy', 'playful'] | ❌ | ['anxious', 'seek_attention', 'happy'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在叫声分析下，宠物类型：哈士奇叫声音描述：哈士奇叫声音描述：轻快短促的汪叫1-2声， |
| 18 | dog | happy(开心) | `你` | ['anxious', 'jealous', 'happy'] | 🔶T3 | ['alert'] | ❌ | 这种宠物类型的声音描述： 轻快短促的汪叫声音描述: 狗狗警觉到异常情况，发出警报叫声音描述: 连续快速的汪汪叫声音描述: |
| 19 | dog | hungry(饥饿) | `啊` | ['happy', 'happy', 'jealous'] | ❌ | ['playful', 'hungry', 'happy'] | 🔶T3 | 这种宠物类型：叫声通常表明宠物感情状态的表现。在叫声通常表明宠物感情状态的表现下，建议检查喂食计划，按量喂食；避免喂人类 |
| 20 | dog | pain(疼痛) | `嘗嘗嘗` | ['anxious', 'playful', 'jealous'] | ❌ | ['hungry', 'anxious'] | ❌ | 这种宠物类型的声音描述：嘗嘗嘗 叫声音描述：感识到焦虑不安，环境变情绪: 主人出现后雷雨天，搬家、新成员加入、雷雨天。建 |
| 21 | dog | playful(玩耍) | `你` | ['anxious', 'jealous', 'happy'] | ❌ | ['hungry', 'excited', 'alert'] | ❌ | 这种叫声通常表明宠物感情的状态。在宠物类型：察识的叫声分析下， Biografia的声音描述：声音描述: 轻快短促的汪叫 |
| 22 | dog | territorial(领地意识) | `哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦哦` | ['happy', 'happy', 'happy'] | ❌ | ['happy', 'hungry', 'content'] | ❌ | 这种叫声通常表明宠物心情愉悦的状态。在宠物类型的情况下，建议满足猫咪的期待，适度互动或喂食，避免过idense刺激。 |

### 4.3 分情绪 E2E 对比
| 情绪(中) | N | RAG Top1Acc | RAG Top3Acc | LLM Top1Acc | LLM Top3Acc |
|---|---|---|---|---|---|
| 警惕 | 2 | 0% | 0% | 0% | 0% |
| 愤怒 | 2 | 0% | 0% | 0% | 0% |
| 焦虑 | 2 | 0% | 0% | 0% | 0% |
| 兴奋 | 2 | 0% | 50% | 0% | 0% |
| 恐惧 | 2 | 0% | 0% | 0% | 0% |
| 开心 | 2 | 0% | 50% | 0% | 50% |
| 饥饿 | 2 | 0% | 0% | 50% | 100% |
| 疼痛 | 2 | 0% | 0% | 0% | 0% |
| 满足 | 1 | 0% | 0% | 0% | 0% |
| 挫败 | 1 | 100% | 100% | 0% | 0% |
| 孤独 | 1 | 0% | 0% | 0% | 0% |
| 问候 | 1 | 0% | 0% | 0% | 0% |
| 玩耍 | 1 | 0% | 0% | 0% | 0% |
| 领地意识 | 1 | 0% | 0% | 0% | 0% |

## 5. ⚠️ 模型需要改进的点 (按优先级)

### P1 [P0] Whisper: Whisper 文本→情绪关键词 Top3 命中率仅 27%
- **根因**: ASR 输出和情绪完全不相关，是噪声；下游关键词匹配等于猜
- **建议解决方案**:
  - ① 改造 Whisper fine-tune 任务为 情绪分类，输出 emotion logits
  - ② 或前端加一个小 CNN 做基频+频谱情绪分类，独立决策

### P2 [P1] RAG (Embedding): RAG Top1 命中率 5% < 35%
- **根因**: Whisper 文本是乱码/噪声，Embedding 后和知识库中文描述差距大；知识库里 cat/dog 的声音描述以 "急促短喵" 等中文为主，ASR 出不来就匹配不上
- **建议解决方案**:
  - ① RAG 查询改为直接用 音频特征 (Audio Embedding, e.g. CLAP) 而不是文本
  - ② 知识库中每个 entry 同时保存 多个 ASR 伪标签的同义词变体 (嗚/汪/喵/hungry...)
  - ③ 查询用 pet_type+breed+场景词 重加权，而不是纯 ASR 文本

### P3 [P1] LLM+RAG 协同: E2E LLM 情绪 Top3 命中率 14%
- **根因**: 上游 Whisper 识别噪声 + RAG 未命中 导致 LLM 输入完全是错的，是典型的 "Garbage In Garbage Out"
- **建议解决方案**:
  - ① 最优先修复 Whisper (改为音频分类，而不是 ASR 识别文字)
  - ② 当 Whisper 文本≤2字时，直接跳过 ASR→RAG 文本路径，改为用 pet_type+breed+音频基频/能量/RMS 规则检索情绪
  - ③ Pipeline 新增一层：声学特征→情绪 规则分类器 (决策树/lightGBM)，与 RAG+LLM 加权 ensemble