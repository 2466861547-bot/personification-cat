"""
推理 Pipeline: 完整的宠物声音翻译流水线
宠物声音 → Whisper ASR → RAG 检索 → LLM 分析 → 生成回复
人类语言 → LLM 意图分析 → 音频合成 → 宠物声音
"""

import os
import sys
import json
from typing import Dict, Optional

from ..audio.audio_processor import AudioProcessor
from ..rag.knowledge_base import PetKnowledgeBase
from ..rag.retriever import PetRetriever
from ..models.whisper_finetune import WhisperFineTuner, WhisperFineTuneConfig
from ..models.llm_finetune import LLMFineTuner, LLMFineTuneConfig
from ..models.audio_generation import AudioGenerator
from ..models.pet_sound_classifier import PetSoundClassifier
from ..agent.pet_agent import PetTranslationAgent
from ..agent.tools import PetTools


class TranslationPipeline:
    """宠物翻译完整流水线"""

    def __init__(
        self,
        whisper_model_path: Optional[str] = None,
        llm_adapter_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        use_rag: bool = True,
        use_agent: bool = True,
    ):
        """
        Args:
            whisper_model_path: Whisper 微调模型路径
            llm_adapter_path: LLM LoRA adapter 路径
            embedding_model: 自定义 Embedding 模型名 (None 则自动检测)
            use_rag: 是否启用 RAG
            use_agent: 是否使用 Agent 编排
        """
        self.audio_processor = AudioProcessor()
        self.whisper = None
        self.llm = None
        self.rag_retriever = None
        self.audio_generator = None
        self.pet_classifier = None
        self.agent = None
        self._custom_embedding_model = embedding_model

        self._init_components(
            whisper_model_path, llm_adapter_path, use_rag, use_agent
        )

    def _init_components(
        self,
        whisper_path: Optional[str],
        llm_path: Optional[str],
        use_rag: bool,
        use_agent: bool,
    ):
        """初始化各组件 (网络断开时优雅降级) — 按业务架构分层打印日志"""

        # ============ 模型服务层开始 ============
        print(f"\n{'='*60}")
        print(f"🧠 【模型服务层】加载模型组件")
        print(f"{'='*60}")

        # 1. Whisper ASR
        print(f"  ┌─ 模块 1/3: Whisper 微调 (宠物声音 ASR)")
        whisper_config = WhisperFineTuneConfig()
        if whisper_path and os.path.exists(whisper_path):
            try:
                self.whisper = WhisperFineTuner(whisper_config)
                self.whisper.load_finetuned(whisper_path)
                print(f"  └─ ✅ Whisper 微调模型加载成功: {os.path.basename(whisper_path)}")
            except Exception as e:
                print(f"  └─ ⚠️  Whisper 微调模型加载失败: {e}")
                self.whisper = None
        else:
            try:
                self.whisper = WhisperFineTuner(whisper_config)
                if not self.whisper.is_ready:
                    print("  └─ ⚠️  Whisper 基础模型未就绪 (网络不可用或未缓存)")
                    print("     pet_to_text 模式需要 Whisper，可尝试:")
                    print("       1. 连接网络后重试")
                    print("       2. 使用已训练好的 Whisper 模型: --whisper_model ./checkpoints/whisper/final")
                    print("       3. 下载 whisper-tiny 模型到本地缓存")
                    print("     text_to_pet / chat 模式仍可正常使用")
                    self.whisper = None
                else:
                    print(f"  └─ ✅ Whisper 基础模型加载成功")
            except Exception as e:
                print(f"  └─ ⚠️  Whisper 加载失败: {e}")
                self.whisper = None

        # 2. LLM
        print(f"  ┌─ 模块 2/3: LLM LoRA 微调 (情绪理解)")
        llm_config = LLMFineTuneConfig()
        if llm_path and os.path.exists(llm_path):
            try:
                self.llm = LLMFineTuner(llm_config, adapter_path=llm_path)
                print(f"  └─ ✅ LLM LoRA adapter 加载成功: {os.path.basename(llm_path)}")
            except Exception as e:
                print(f"  └─ ⚠️  LLM adapter 加载失败: {e}")
                self.llm = None
        else:
            try:
                self.llm = LLMFineTuner(llm_config)
                if not self.llm.is_ready:
                    print("  └─ ⚠️  LLM 基础模型未就绪 (网络不可用或未缓存)")
                    self.llm = None
                else:
                    print(f"  └─ ✅ LLM 基础模型加载成功")
            except Exception as e:
                print(f"  └─ ⚠️  LLM 加载失败: {e}")
                self.llm = None

        # 3. Bark 音频生成
        print(f"  ┌─ 模块 3/4: Bark 音频生成 (宠物声音合成 + 人类声纹克隆)")
        try:
            self.audio_generator = AudioGenerator()
            if self.audio_generator.is_ready:
                print(f"  └─ ✅ Bark 模型加载成功")
            else:
                print(f"  └─ ⚠️  Bark 不可用，降级为模拟音频生成")
        except Exception as e:
            print(f"  └─ ⚠️  音频生成器加载失败: {e}")
            self.audio_generator = None

        # 4. 宠物声音分类器 (替代 Whisper 处理宠物声音)
        print(f"  ┌─ 模块 4/4: 宠物声音分类器 (声学特征 + 随机森林)")
        classifier_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "checkpoints",
            "pet_classifier",
        )
        if os.path.isdir(classifier_path):
            try:
                self.pet_classifier = PetSoundClassifier(model_dir=classifier_path)
                if self.pet_classifier.is_ready:
                    print(f"  └─ ✅ 宠物声音分类器加载成功")
                else:
                    print(f"  └─ ⚠️  分类器目录存在但加载失败，将仅使用 Whisper")
                    self.pet_classifier = None
            except Exception as e:
                print(f"  └─ ⚠️  宠物声音分类器加载失败: {e}")
                self.pet_classifier = None
        else:
            print(f"  └─ ℹ️  未训练宠物声音分类器")
            print(f"     训练命令: python scripts/train_pet_classifier.py")
            print(f"     未训练时将使用 Whisper (效果有限)")
            self.pet_classifier = None

        print(f"  {'─'*60}")
        # ============ 模型服务层结束 ============

        # ============ RAG 知识检索层开始 ============
        print(f"\n{'='*60}")
        print(f"📚 【RAG 知识检索层】构建知识库索引")
        print(f"{'='*60}")

        if use_rag:
            from ..rag.retriever import EmbeddingConfig
            print(f"  ┌─ 加载宠物知识库 (预置知识)")
            kb = PetKnowledgeBase()
            print(f"  ├─ 初始化 Embedding 模型 + ChromaDB 向量库")
            if self._custom_embedding_model:
                config = EmbeddingConfig()
                config.high_quality_model = self._custom_embedding_model
                config.lightweight_model = self._custom_embedding_model
                self.rag_retriever = PetRetriever(
                    knowledge_base=kb, embedding_config=config
                )
            else:
                self.rag_retriever = PetRetriever(knowledge_base=kb)
            print(f"  ├─ 向量化文档并构建向量索引...")
            rag_ok = self.rag_retriever.build_index()
            if not rag_ok:
                print("  └─ ⚠️  RAG 索引构建失败 (Embedding 模型不可用)")
                print("     RAG 检索不可用，将使用 LLM 直接分析 (无知识库增强)")
                print("     解决方法: 下载 Embedding 模型到本地缓存后重试")
            else:
                print(f"  └─ ✅ RAG 知识检索层就绪")
        else:
            print(f"  └─ ⏭️  RAG 已禁用 (--no-rag)，跳过知识检索")

        print(f"  {'─'*60}")
        # ============ RAG 知识检索层结束 ============

        # ============ LangChain Agent 编排层开始 ============
        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】初始化编排组件")
        print(f"{'='*60}")

        if use_agent and self.llm:
            print(f"  ┌─ 注册 Agent 工具集:")
            tools = PetTools(
                retriever=self.rag_retriever,
                audio_generator=self.audio_generator,
                llm=self.llm,
            )
            tool_names = []
            if self.rag_retriever and self.rag_retriever.is_ready:
                tool_names.append("知识检索 Tool")
            if self.audio_generator:
                tool_names.append("声音生成 Tool")
            tool_names.extend(["声音分析 Tool", "情绪解读 Tool"])
            for name in tool_names:
                print(f"  │     ✅ {name}")
            try:
                self.agent = PetTranslationAgent(tools=tools, llm=self.llm)
                print(f"  └─ ✅ LangChain Agent 编排层就绪 (工具数: {len(tool_names)})")
            except Exception as e:
                print(f"  └─ ⚠️  Agent 初始化失败: {e} → 降级为直接 LLM 调用")
                self.agent = None
        else:
            if not self.llm:
                print(f"  └─ ⚠️  LLM 不可用，跳过 Agent 编排层")
            else:
                print(f"  └─ ⏭️  use_agent=False，跳过 Agent 编排层，降级为 LLM 直调")
            self.agent = None

        print(f"  {'─'*60}")
        # ============ LangChain Agent 编排层结束 ============

        # 初始化完成总结
        print(f"\n{'='*60}")
        print(f"✅ 系统初始化完成")
        print(f"{'='*60}")
        components = [
            ("Whisper ASR", "✅" if self.whisper else "❌"),
            ("LLM 情绪解读", "✅" if self.llm else "❌"),
            ("RAG 知识检索", "✅" if self.rag_retriever and self.rag_retriever.is_ready else "❌"),
            ("Audio 声音生成", "✅" if self.audio_generator else "❌"),
            ("宠物声音分类器", "✅" if self.pet_classifier else "⏭️"),
            ("Agent 编排", "✅" if self.agent else "⏭️"),
        ]
        for name, status in components:
            print(f"  {status} {name}")
        print()

    def pet_sound_to_text(
        self,
        audio_path: str,
        pet_type: str = "cat",
        breed: str = "通用",
    ) -> Dict:
        """宠物声音 → 人类语言 — 使用分类器 (优先) 或 Whisper (fallback)"""
        result = {
            "input_audio": audio_path,
            "pet_type": pet_type,
            "rag_details": [],
        }

        # ============ LangChain Agent 编排层 Step 1: 声音分析 ============
        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 宠物声音 → 人类语言")
        print(f"{'='*60}")
        print(f"  Step 1/3: 声音分析")

        # 优先使用宠物声音分类器 (如果已就绪)
        if self.pet_classifier and self.pet_classifier.is_ready:
            print(f"    ↓ 使用: 宠物声音分类器 (声学特征 + 随机森林)")
            classification = self.pet_classifier.predict(audio_path)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["pet_classification"] = classification
            result["whisper_transcription"] = description  # 用描述文本替代 Whisper 输出
            result["emotion_confidence"] = confidence
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f})")
            print(f"    ↑ 描述: {description}")
        elif self.whisper:
            print(f"    ↓ 使用: Whisper ASR (fallback)")
            whisper_result = self.whisper.inference(audio_path)
            result["whisper_transcription"] = whisper_result
            print(f"    ↑ Whisper 输出: '{whisper_result}'")
        else:
            result["whisper_transcription"] = "无法识别(无可用模型)"
            print(f"    ↑ ⚠️  无可用模型")

        # ============ LangChain Agent 编排层 Step 2: 知识检索 ============
        print(f"  Step 2/3: 知识检索")
        search_query = result["whisper_transcription"]
        print(f"    ↓ 输入: '{search_query}' → Embedding + ChromaDB (RAG 知识检索层)")

        # 2. RAG 检索 (RAG 知识检索层)
        if self.rag_retriever and self.rag_retriever.is_ready:
            search_detail = self.rag_retriever.search_with_details(
                f"{pet_type} {search_query} 情绪 含义 建议"
            )
            rag_results = search_detail.get("results", [])
            result["rag_details"] = rag_results

            print(f"    ↑ 输出: 命中 {len(rag_results)} 条相关知识")
            for i, r in enumerate(rag_results, 1):
                meta = r.get("metadata", {})
                score = r.get("score", 0)
                text_preview = r.get("text", "")[:60]
                print(f"      [{i}] 相关度={score:.3f} | {meta.get('pet_type','')}/{meta.get('breed','')}/{meta.get('emotion','')}")
                print(f"           {text_preview}...")
        else:
            print(f"    ↑ 输出: RAG 不可用，跳过知识检索")
            rag_results = []

        # ============ LangChain Agent 编排层 Step 3: 情绪解读 ============
        print(f"  Step 3/3: 情绪解读")
        print(f"    ↓ 输入: 声音分析结果 + RAG上下文 → LLM LoRA (模型服务层)")

        # 3. LLM 分析
        if self.agent:
            analysis = self.agent.analyze_sound(
                sound_description=search_query,
                pet_type=pet_type,
                breed=breed,
            )
            result["analysis"] = analysis.get("analysis", str(analysis))
        elif self.llm:
            rag_context = ""
            if self.rag_retriever and self.rag_retriever.is_ready:
                rag_context = self.rag_retriever.get_context_for_llm(
                    f"{pet_type} {search_query}"
                )

            system_prompt = "你是一个宠物行为学专家。"
            user_input = (
                f"宠物类型: {pet_type}, 品种: {breed}\n"
                f"声音描述: {search_query}\n\n"
            )
            if rag_context:
                user_input += f"知识库参考:\n{rag_context}\n\n"
            user_input += "请分析情绪和需求。"

            result["analysis"] = self.llm.inference(user_input, system_prompt)
        else:
            result["analysis"] = "LLM 未加载"

        output_len = len(result["analysis"]) if result["analysis"] else 0
        print(f"    ↑ 输出: 情绪解读完成 ({output_len} 字)")
        print(f"  {'─'*60}")

        return result

    def pet_sound_to_human_sound(
        self,
        audio_path: str,
        target_voice: str = "default",
        pet_type: str = "cat",
        breed: str = "通用",
        output_path: str = "",
    ) -> Dict:
        """宠物声音 → 人类语言 → 人类声音 (声纹克隆)"""
        result = {
            "input_audio": audio_path,
            "target_voice": target_voice,
            "pet_type": pet_type,
        }

        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 宠物声音 → 人类声音 (声纹克隆)")
        print(f"{'='*60}")

        # Step 1: 宠物声音 → 情绪分类
        print(f"  Step 1/3: 宠物声音情绪分类")
        if self.pet_classifier and self.pet_classifier.is_ready:
            classification = self.pet_classifier.predict(audio_path)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["emotion"] = emotion
            result["emotion_confidence"] = confidence
            result["description"] = description
            print(f"    ↓ 使用: 宠物声音分类器")
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f})")
            print(f"    ↑ 中文描述: {description}")
        elif self.whisper:
            print(f"    ↓ 使用: Whisper ASR (fallback)")
            whisper_text = self.whisper.inference(audio_path)
            result["whisper_text"] = whisper_text
            description = f"宠物发出了 '{whisper_text}' 的声音"
            result["description"] = description
            print(f"    ↑ Whisper: '{whisper_text}'")
        else:
            description = "无法识别宠物声音"
            result["description"] = description
            print(f"    ↑ ⚠️  无可用模型")

        # Step 2: LLM 生成拟人化描述 (更自然的人类语言)
        print(f"  Step 2/3: LLM 生成拟人化描述")
        if self.llm:
            system_prompt = "你是一个宠物翻译专家。请将宠物的情绪/声音描述转化为自然、口语化的人类语言，让人能听懂宠物想表达什么。"
            user_input = (
                f"宠物类型: {pet_type}\n"
                f"品种: {breed}\n"
                f"情绪: {result.get('emotion', 'unknown')}\n"
                f"描述: {result['description']}\n\n"
                f"请将这句话转化为一句自然的人话，让人能听懂宠物的意思。"
            )
            human_text = self.llm.inference(user_input, system_prompt)
            result["human_text"] = human_text
            print(f"    ↑ 拟人化文本: {human_text[:100]}...")
        else:
            human_text = result["description"]
            result["human_text"] = human_text
            print(f"    ↑ LLM 未加载，使用原始描述")

        # Step 3: 文字 → 人类声音 (声纹克隆)
        print(f"  Step 3/3: 文字 → 人类声音 (目标声线: {target_voice})")
        if self.audio_generator:
            if not output_path:
                os.makedirs("./output", exist_ok=True)
                output_path = "./output/pet_to_human_cloned.wav"

            audio = self.audio_generator.text_to_human_sound(
                text=human_text,
                target_voice=target_voice,
                output_path=output_path,
            )
            result["status"] = "success"
            result["output_path"] = output_path
            if not self.audio_generator.is_ready:
                result["note"] = "使用模拟音频生成 (Bark 模型不可用)"
        else:
            result["status"] = "no_audio_generator"
            result["note"] = "音频生成器不可用"

        print(f"  {'─'*60}")
        return result

    def human_text_to_pet_sound(
        self,
        text: str,
        target_pet: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """人类语言 → 宠物声音"""
        result = {
            "input_text": text,
            "target_pet": target_pet,
        }

        if self.agent:
            translation = self.agent.translate_to_pet(
                human_text=text,
                target_pet=target_pet,
                output_path=output_path,
            )
            result.update(translation)
        elif self.audio_generator:
            # 自动生成默认输出路径
            if not output_path:
                os.makedirs("./output", exist_ok=True)
                output_path = f"./output/{target_pet}_sound.wav"

            audio = self.audio_generator.human_speech_to_pet_sound(
                speech_text=text,
                target_pet=target_pet,
                output_path=output_path,
            )
            result["status"] = "success"
            result["output_path"] = output_path
            if not self.audio_generator.is_ready:
                result["note"] = "使用模拟音频生成 (Bark 模型不可用，降级模式)"
        else:
            result["status"] = "no_audio_generator"
            result["note"] = "音频生成器不可用"

        return result

    def human_sound_to_human_sound(
        self,
        audio_path: str,
        target_voice: str = "default",
        output_path: str = "",
    ) -> Dict:
        """人类声音 → 人类语言 → 人类声音 (声纹克隆)"""
        result = {
            "input_audio": audio_path,
            "target_voice": target_voice,
        }

        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 人类声音 → 人类声音 (声纹克隆)")
        print(f"{'='*60}")
        print(f"  Step 1/3: 人类声音 → 文字 (Whisper ASR)")
        if self.whisper:
            text_result = self.whisper.inference(audio_path)
            result["recognized_text"] = text_result
            print(f"    ↑ 识别结果: '{text_result}'")
        else:
            text_result = "无法识别(Whisper 未加载)"
            result["recognized_text"] = text_result
            print(f"    ↑ ⚠️  Whisper 未加载，跳过 ASR")

        print(f"  Step 2/3: LLM 意图分析 (可选)")
        if self.llm and text_result and "无法识别" not in text_result:
            system_prompt = "你是一个语音助手。请简要分析用户这句话的意图和情绪，然后原样返回用户的核心文本用于语音合成。格式: [意图:xxx] [情绪:xxx] [文本:xxx]"
            analysis = self.llm.inference(text_result, system_prompt)
            result["llm_analysis"] = analysis
            print(f"    ↑ LLM 分析: {analysis[:100]}...")
            # 尝试提取核心文本用于 TTS
            import re
            text_match = re.search(r'文本[：:](.*)', analysis)
            if text_match:
                text_for_tts = text_match.group(1).strip()
            else:
                text_for_tts = text_result
        else:
            text_for_tts = text_result
            print(f"    ↑ ⚠️  LLM 未加载，直接使用识别文本")

        print(f"  Step 3/3: 文字 → 人类声音 (目标声线: {target_voice})")
        if self.audio_generator and text_for_tts:
            if not output_path:
                os.makedirs("./output", exist_ok=True)
                output_path = "./output/human_cloned.wav"

            audio = self.audio_generator.text_to_human_sound(
                text=text_for_tts,
                target_voice=target_voice,
                output_path=output_path,
            )
            result["status"] = "success"
            result["output_path"] = output_path
            if not self.audio_generator.is_ready:
                result["note"] = "使用模拟音频生成 (Bark 模型不可用，降级模式)"
        else:
            result["status"] = "no_audio_generator"
            result["note"] = "音频生成器不可用"

        print(f"  {'─'*60}")
        return result

    def human_sound_to_pet_sound(
        self,
        audio_path: str,
        target_pet: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """人类声音 → 人类语言 → 宠物声音"""
        result = {
            "input_audio": audio_path,
            "target_pet": target_pet,
        }

        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 人类声音 → 宠物声音")
        print(f"{'='*60}")
        print(f"  Step 1/3: 人类声音 → 文字 (Whisper ASR)")
        if self.whisper:
            text_result = self.whisper.inference(audio_path)
            result["recognized_text"] = text_result
            print(f"    ↑ 识别结果: '{text_result}'")
        else:
            text_result = "无法识别(Whisper 未加载)"
            result["recognized_text"] = text_result
            print(f"    ↑ ⚠️  Whisper 未加载")

        print(f"  Step 2/3: LLM 意图分析 → 映射宠物情绪")
        if self.llm and text_result and "无法识别" not in text_result:
            # 复用文本转宠物声音的逻辑
            human_result = self.human_text_to_pet_sound(
                text=text_result,
                target_pet=target_pet,
                output_path=output_path,
            )
            result.update(human_result)
            print(f"    ↑ 分析完成，目标宠物: {target_pet}")
        elif self.audio_generator:
            # 降级: 用简单意图分析
            if not output_path:
                os.makedirs("./output", exist_ok=True)
                output_path = f"./output/{target_pet}_from_human.wav"

            audio = self.audio_generator.human_speech_to_pet_sound(
                speech_text=text_result,
                target_pet=target_pet,
                output_path=output_path,
            )
            result["status"] = "success"
            result["output_path"] = output_path
        else:
            result["status"] = "no_audio_generator"
            result["note"] = "音频生成器不可用"

        print(f"  {'─'*60}")
        return result

    def chat(self, user_input: str) -> str:
        """对话模式"""
        if self.agent:
            return self.agent.chat(user_input)
        elif self.llm:
            return self.llm.inference(user_input)
        else:
            return "系统未初始化"

    def batch_analyze(
        self,
        audio_dir: str,
        pet_type: str = "cat",
    ) -> list:
        """批量分析"""
        results = []
        if not os.path.isdir(audio_dir):
            return results

        for filename in os.listdir(audio_dir):
            if filename.endswith((".wav", ".mp3", ".flac")):
                audio_path = os.path.join(audio_dir, filename)
                result = self.pet_sound_to_text(audio_path, pet_type)
                results.append(result)

        return results
