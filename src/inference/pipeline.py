"""
推理 Pipeline: 完整的宠物声音翻译流水线
三层分类器架构:
  Layer 1: LLMAcousticClassifier (LLM 声学特征分类, 首选)
  Layer 2: PetSoundClassifier (声学特征 + 随机森林, 离线备选)
  Layer 3: Whisper ASR (人类语音 fallback, 保底)
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
from ..models.pet_sound_classifier import PetSoundClassifier, LLMAcousticClassifier, FusionPetClassifier, detect_environment
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
        self.pet_classifier = None       # Layer 2: ML 分类器
        self.llm_classifier = None       # Layer 1: LLM 声学分类器
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

        # 4. 融合宠物声音分类器 (自动检测环境选择策略)
        env_info = detect_environment()
        strategy_desc = env_info["description"]
        print(f"  ┌─ 模块 4/5: 融合宠物声音分类器")
        print(f"  │  环境: {strategy_desc}")
        try:
            self.fusion_classifier = FusionPetClassifier(llm_model=self.llm)
            self.llm_classifier = self.fusion_classifier  # 兼容旧代码引用
            if self.llm_classifier.is_ready:
                print(f"  └─ ✅ 融合分类器就绪 (策略: {env_info['strategy']})")
            else:
                print(f"  └─ ⚠️  融合分类器未就绪")
        except Exception as e:
            print(f"  └─ ⚠️  融合分类器初始化失败: {e}")
            self.fusion_classifier = None
            self.llm_classifier = None

        # 5. Layer 2: 声学特征 + 随机森林分类器 (备选)
        print(f"  ┌─ 模块 5/5: 声学特征 + 随机森林分类器 (Layer 2 - 备选)")
        # pipeline.py 在 src/inference/ 下，需要往上 3 级到项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        classifier_path = os.path.join(project_root, "checkpoints", "pet_classifier")
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

        # 融合分类器: FusionPetClassifier (自动选择策略)
        if self.fusion_classifier and self.fusion_classifier.is_ready:
            env_info = detect_environment()
            strategy = env_info["strategy"]
            print(f"    ↓ 使用: 融合分类器 (策略: {strategy})")
            classification = self.fusion_classifier.classify(audio_path, pet_type)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["pet_classification"] = classification
            result["whisper_transcription"] = description
            result["emotion_confidence"] = confidence
            method = classification.get("method", "fusion")
            fusion_mode = classification.get("fusion_mode", "unknown")
            reasoning = classification.get("llm_reasoning", "")
            species_info = classification.get("species_detection", {})
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f}) [方法: {method}]")
            if fusion_mode != "unknown":
                print(f"    ↑ 融合模式: {fusion_mode}")
            if species_info:
                print(f"    ↑ 物种: {species_info.get('species', '?')} (置信度: {species_info.get('confidence', 0):.2f})")
            if reasoning:
                print(f"    ↑ 推理: {reasoning[:80]}")
            print(f"    ↑ 描述: {description}")
        elif self.pet_classifier and self.pet_classifier.is_ready:
            print(f"    ↓ 使用: 声学特征 + 随机森林 (Layer 2 - 备选)")
            classification = self.pet_classifier.predict(audio_path)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["pet_classification"] = classification
            result["whisper_transcription"] = description
            result["emotion_confidence"] = confidence
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f})")
            print(f"    ↑ 描述: {description}")
        elif self.whisper:
            print(f"    ↓ 使用: Whisper ASR (Layer 3 - 保底)")
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
        use_llm_polish: bool = True,
        reference_audio: str = "",
        reference_text: str = "",
    ) -> Dict:
        """宠物声音 → 人类语言 → 人类声音 (声纹克隆)

        Args:
            use_llm_polish: 是否使用 LLM 润色模板文本 (默认 True)
            reference_audio: 参考音频路径 (3-15秒)，提供则使用 CosyVoice 真实声纹克隆
            reference_text: 参考音频的文字内容 (用于 zero-shot 克隆)
        """
        result = {
            "input_audio": audio_path,
            "target_voice": target_voice,
            "pet_type": pet_type,
        }

        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 宠物声音 → 人类声音 (声纹克隆)")
        print(f"{'='*60}")

        # Step 1: 宠物声音 → 情绪分类 (融合架构)
        print(f"  Step 1/3: 宠物声音情绪分类")
        if self.fusion_classifier and self.fusion_classifier.is_ready:
            env_info = detect_environment()
            print(f"    ↓ 使用: 融合分类器 (策略: {env_info['strategy']})")
            classification = self.fusion_classifier.classify(audio_path, pet_type)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["emotion"] = emotion
            result["emotion_confidence"] = confidence
            result["description"] = description
            result["pet_classification"] = classification
            method = classification.get("method", "fusion")
            fusion_mode = classification.get("fusion_mode", "unknown")
            species_info = classification.get("species_detection", {})
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f}) [方法: {method}]")
            if fusion_mode != "unknown":
                print(f"    ↑ 融合模式: {fusion_mode}")
            if species_info:
                print(f"    ↑ 物种: {species_info.get('species', '?')} (置信度: {species_info.get('confidence', 0):.2f})")
            print(f"    ↑ 中文描述: {description}")
        elif self.pet_classifier and self.pet_classifier.is_ready:
            print(f"    ↓ 使用: 声学特征 + 随机森林 (Layer 2)")
            classification = self.pet_classifier.predict(audio_path)
            emotion = classification["emotion"]
            confidence = classification["confidence"]
            description = classification["description"]
            result["emotion"] = emotion
            result["emotion_confidence"] = confidence
            result["description"] = description
            print(f"    ↑ 情绪: {emotion} (置信度: {confidence:.2f})")
            print(f"    ↑ 中文描述: {description}")
        elif self.whisper:
            print(f"    ↓ 使用: Whisper ASR (Layer 3 - 保底)")
            whisper_text = self.whisper.inference(audio_path)
            result["whisper_text"] = whisper_text
            description = f"宠物发出了 '{whisper_text}' 的声音"
            result["description"] = description
            print(f"    ↑ Whisper: '{whisper_text}'")
        else:
            description = "无法识别宠物声音"
            result["description"] = description
            print(f"    ↑ ⚠️  无可用模型")

        # Step 2: 生成拟人化描述 (模板 + LLM 润色 混合方案)
        print(f"  Step 2/3: 生成拟人化描述 (混合方案)")
        emotion = result.get('emotion', 'unknown')
        description = result.get('description', '')
        pet_names = {"cat": "猫咪", "dog": "狗狗", "bird": "鸟", "rabbit": "兔子"}
        pet_cn = pet_names.get(pet_type, pet_type)
        print(f"    ↑ 宠物类型: {pet_cn} | 品种: {breed} | 情绪: {emotion}")

        # 2.1 模板生成基础文本 (100% 可靠)
        template_text = self._template_anthropomorphic(
            pet_type=pet_type,
            breed=breed,
            emotion=emotion,
            description=description,
        )
        print(f"    ↑ [模板] 基础文本: {template_text}")

        # 2.2 LLM 润色 (可选，失败则回退模板)
        if use_llm_polish and self.llm and self.llm.is_ready:
            print(f"    ↑ [LLM] 正在润色...")
            human_text = self._llm_polish_text(
                template_text=template_text,
                pet_type=pet_type,
                breed=breed,
                emotion=emotion,
            )
            if human_text != template_text:
                print(f"    ↑ [润色] 润色文本: {human_text}")
                print(f"    ↑ ✅ 使用 LLM 润色版本")
            else:
                print(f"    ↑ ℹ️ LLM 未改进，使用模板文本")
        else:
            human_text = template_text
            if not use_llm_polish:
                print(f"    ↑ ℹ️ LLM 润色已禁用 (--no-llm-polish)")
            else:
                print(f"    ↑ ℹ️ LLM 不可用，使用模板文本")

        result["human_text"] = human_text
        result["human_text_source"] = "llm_polished" if human_text != template_text else "template"
        print(f"    ↑ 最终拟人化文本: {human_text}")

        # Step 3: 文字 → 人类声音 (声纹克隆)
        cosyvoice_info = f"CosyVoice声纹克隆 (参考: {reference_audio})" if reference_audio else target_voice
        print(f"  Step 3/3: 文字 → 人类声音 ({cosyvoice_info})")
        if self.audio_generator:
            if not output_path:
                os.makedirs("./output", exist_ok=True)
                output_path = "./output/pet_to_human_cloned.wav"

            audio = self.audio_generator.text_to_human_sound(
                text=human_text,
                target_voice=target_voice,
                output_path=output_path,
                reference_audio=reference_audio or None,
                reference_text=reference_text or None,
            )
            result["status"] = "success"
            result["output_path"] = output_path
            if reference_audio:
                result["voice_cloning"] = "cosyvoice"
            elif not self.audio_generator.is_ready:
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
        reference_audio: str = "",
        reference_text: str = "",
    ) -> Dict:
        """人类声音 → 人类语言 → 人类声音 (声纹克隆)

        Args:
            reference_audio: 参考音频路径 (3-15秒)，提供则使用 CosyVoice 真实声纹克隆
            reference_text: 参考音频的文字内容 (用于 zero-shot 克隆)
        """
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
                reference_audio=reference_audio or None,
                reference_text=reference_text or None,
            )
            result["status"] = "success"
            result["output_path"] = output_path
            if reference_audio:
                result["voice_cloning"] = "cosyvoice"
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

    @staticmethod
    def _template_anthropomorphic(
        pet_type: str,
        breed: str,
        emotion: str,
        description: str,
    ) -> str:
        """模板生成拟人化文本 — 丰富多品种、多情绪、口语化"""

        pet_cn = {"cat": "猫咪", "dog": "狗狗", "bird": "小鸟", "rabbit": "兔兔"}.get(pet_type, pet_type)
        pronoun = "它" if pet_type not in ("cat", "dog") else ("猫猫" if pet_type == "cat" else "狗狗")

        # 品种特定的口语习惯
        breed_hints = {
            "橘猫": ["橘胖", "小胖橘"],
            "英短": ["英短", "蓝胖子"],
            "美短": ["美短", "花纹"],
            "布偶": ["布偶", "小仙女"],
            "狸花": ["狸花", "小狸"],
            "加菲猫": ["加菲", "胖喵"],
            "俄罗斯蓝猫": ["俄蓝", "蓝猫"],
            "柯基": ["柯基", "小短腿"],
            "金毛": ["金毛", "大暖男"],
            "泰迪": ["泰迪", "小机灵"],
            "柴犬": ["柴犬", "小柴"],
            "哈士奇": ["二哈", "撒手没"],
            "比熊": ["比熊", "棉花糖"],
            "边牧": ["边牧", "小天才"],
            "阿拉斯加": ["阿拉斯加", "阿拉"],
            "博美": ["博美", "小狐狸"],
            "雪纳瑞": ["雪纳瑞", "小老头"],
            "拉布拉多": ["拉布拉多", "拉拉"],
        }
        hints = breed_hints.get(breed, [breed])

        import random

        # 按宠物类型 + 情绪 组合模板
        if pet_type == "cat":
            templates = {
                "hungry": [
                    f"{hints[0]}：铲屎官，我的饭碗空啦~快喂我！",
                    f"{hints[0]}用爪子扒拉你的腿：我饿了啦，别装没看见！",
                    f"喵~ 主人，你闻到我肚子咕咕叫了吗？",
                    f"{hints[0]}跳上餐桌：今天的饭点好像有点晚哦？",
                    f"猫咪蹲在食盆前久久不肯走：主人...难道忘了什么吗？",
                ],
                "happy": [
                    f"喵呜~ 今天阳光好好，主人摸摸我的头嘛~",
                    f"{hints[0]}翻出肚皮：看！我给你展示我的肚皮！",
                    f"呼噜呼噜~ 被主人摸得好舒服呀~",
                    f"{hints[0]}跳起来给你一个贴面礼：主人最棒啦！",
                    f"尾巴摇成螺旋桨：今天心情特别好，想跟主人玩！",
                ],
                "alert": [
                    f"嘘~ 我好像听到了什么声音，让我去看看！",
                    f"{hints[0]}竖起耳朵盯着窗外：那是什么？",
                    f"小声喵呜：主人，你有没有听到奇怪的声音？",
                    f"{hints[0]}弓起背：有情况！我来保护你！",
                    f"别出声~ 我在警戒中，发现可疑目标了！",
                ],
                "seek_attention": [
                    f"喵~ 你已经看手机十分钟了，理理我嘛~",
                    f"{hints[0]}把脑袋塞到你手里：别玩手机啦，摸摸我！",
                    f"在你脚边转圈圈：主人主人，陪我玩逗猫棒！",
                    f"用爪子拍拍你的书：看书哪有我重要呀！",
                    f"{hints[0]}跳到你腿上：嗯~ 这样就对了嘛。",
                ],
                "angry": [
                    f"哈！别碰我的尾巴，我现在很生气！",
                    f"{hints[0]}甩了甩尾巴：再摸我就咬你了哦！",
                    f"哼！铲屎官今天做了坏事，我不理你了！",
                    f"炸毛中...别过来，我很凶的！",
                    f"{hints[0]}发出低沉的呜呜声：我在生气，快道歉！",
                ],
                "fearful": [
                    f"呜...那个东西好可怕，主人抱抱我好不好？",
                    f"{hints[0]}躲到沙发后面：我不出去...好害怕。",
                    f"小声发颤的喵：主人...你会保护我的对吗？",
                    f"尾巴炸成毛球：那个东西走开了吗？",
                    f"紧紧扒住你的腿：我...我不太敢动...",
                ],
                "content": [
                    f"呼噜~ 这样晒太阳好舒服呀~",
                    f"{hints[0]}慵懒地伸懒腰：嗯~ 完美的一天。",
                    f"眯着眼享受按摩：主人的手艺越来越好了~",
                    f"蜷在你腿上打呼噜：这样就好，不要动哦。",
                    f"满足地舔爪子：今天过得真惬意呀。",
                ],
                "pain": [
                    f"呜...主人，我好像有点不舒服...",
                    f"{hints[0]}蜷缩着不动：别碰我...有点痛。",
                    f"小声呻吟：主人...带我去看医生好不好？",
                    f"眼神哀怨地看着你：我...我好像生病了。",
                    f"慢吞吞地走到你身边：主人...要抱抱。",
                ],
                "lonely": [
                    f"喵...你什么时候才回来呀...",
                    f"{hints[0]}望着门口发呆：一个人在家好无聊。",
                    f"趴在你的拖鞋上：主人...快点回来。",
                    f"看着窗外的夕阳：今天又要一个人睡了呢。",
                    f"无精打采地叫了一声：有人在家吗...",
                ],
                "excited": [
                    f"哇！主人拿出了逗猫棒！好开心！",
                    f"{hints[0]}上蹿下跳：看我的三连跳！",
                    f"尾巴摇得飞快：有好玩的啦！有好玩的啦！",
                    f"飞扑！精准命中逗猫棒！",
                    f"喵喵喵~ 今天太开心啦！",
                ],
                "playful": [
                    f"来呀来呀！我们玩躲猫猫~",
                    f"{hints[0]}扒拉逗猫棒：看我抓到你！",
                    f"弓起身子蹦跶：我现在是一只小老虎！",
                    f"用爪子拍拍你的手：陪我玩嘛~",
                    f"{hints[0]}摆出攻击姿态：嘿！吃我一招！",
                ],
                "relaxed": [
                    f"嗯~ 就这样躺着，不要打扰我。",
                    f"{hints[0]}打了个哈欠：今天就是咸鱼的一天。",
                    f"闭着眼：阳光的味道真好闻。",
                    f"换了个姿势继续躺：舒服~",
                    f"发出小呼噜声：zZ~ 别打扰我的美梦。",
                ],
                "curious": [
                    f"咦？那是什么东西？让我闻闻。",
                    f"{hints[0]}歪着脑袋：主人你在做什么呢？",
                    f"小心翼翼地凑过去：这个...可以吃吗？",
                    f"用爪子戳了戳：咦~ 会动的！",
                    f"眼睛睁得圆圆的：新世界解锁了！",
                ],
                "anxious": [
                    f"主人...你今天什么时候回家呀？",
                    f"在门口来回踱步：还不回来吗...",
                    f"不停地舔毛：有点紧张呢。",
                    f"小声叹气：算了，先睡一觉吧。",
                ],
                "frustrated": [
                    f"哈！这个玩具怎么拆不开呀！",
                    f"{hints[0]}甩了甩玩具：哼，我不玩了！",
                    f"小声嘟囔：有点生气，但又不想表现出来。",
                ],
                "sad": [
                    f"喵...主人好像不太开心...",
                    f"安静地走到你身边：要不...我陪你坐会儿？",
                    f"用头蹭了蹭你的手：一切都会好起来的。",
                ],
                "greeting": [
                    f"主人终于回来啦！好想你呀~",
                    f"{hints[0]}跑到门口迎接你：喵~ 你终于回来了！",
                    f"尾巴绕着你的腿转圈圈：今天也辛苦啦！",
                    f"用头蹭你的腿：欢迎回家，铲屎官！",
                ],
                "territorial": [
                    f"这是我的地盘！闲人免进！",
                    f"{hints[0]}挡在门口：想进来？先过我这关！",
                    f"小声呜呜：这里是我的领地。",
                ],
                "confused": [
                    f"歪着头：主人，你在说什么？",
                    f"{hints[0]}一脸懵：我...我没太懂。",
                    f"眨了眨眼：这个指令是什么意思呀？",
                ],
                "jealous": [
                    f"哼！你在摸谁？我才是最重要的！",
                    f"{hints[0]}挤到你和别的宠物中间：看我！",
                    f"小声嘟囔：我才不要别人分走你的爱。",
                ],
            }
        elif pet_type == "dog":
            templates = {
                "hungry": [
                    f"汪！主人，我的饭呢？该开饭啦！",
                    f"{hints[0]}叼着饭盆跑过来：看！我的饭碗！",
                    f"舔了舔你的手：主人，别忘了我哦~",
                    f"{hints[0]}趴在食盆前不肯走：肚子咕咕叫呢。",
                    f"眼神专注地看着你手里的东西：那个...是给我的吗？",
                ],
                "happy": [
                    f"汪汪汪！主人回来啦！好想你哦！",
                    f"{hints[0]}摇着尾巴转圈：看！我等你好久了！",
                    f"给你一个热情的贴面礼：舔舔舔~",
                    f"蹦跶起来：今天也要一起玩球哦！",
                    f"尾巴摇成电风扇：开心！开心！开心！",
                ],
                "alert": [
                    f"汪！有人来了！我去看看！",
                    f"{hints[0]}挡在你面前：主人，有情况我保护你！",
                    f"竖起耳朵盯着门口：别担心，我在警戒。",
                    f"轻声呜呜：那个...好像有陌生人。",
                    f"大声汪汪：谁在外面？报上名来！",
                ],
                "seek_attention": [
                    f"主人主人主人！看我！看我！",
                    f"{hints[0]}用爪子扒拉你：别玩手机了陪我玩！",
                    f"叼来你的拖鞋：来嘛~ 我们出去玩！",
                    f"坐到你腿上：这样对了嘛，摸摸我。",
                    f"盯着你看了很久很久：铲屎官，我很可爱你知道吗？",
                ],
                "angry": [
                    f"汪！别碰我的玩具！",
                    f"{hints[0]}低吼着：我现在很生气，别惹我。",
                    f"哼！我不理你了！（但还是偷偷看你）",
                    f"耳朵背过去：再靠近我就...汪汪了！",
                    f"发出呜呜的警告：这个东西是我的！",
                ],
                "fearful": [
                    f"呜...主人，我有点怕...",
                    f"{hints[0]}躲在你身后：那个东西好可怕。",
                    f"夹着尾巴：我...我不是故意的...",
                    f"小声呜咽：主人，你不会怪我吧？",
                    f"紧紧贴着你的腿：有你在就不怕了。",
                ],
                "content": [
                    f"嗯~ 这样趴着晒太阳好舒服呀。",
                    f"{hints[0]}打了个大哈欠：今天真是好日子。",
                    f"摇着尾巴闭眼享受：主人，摸摸我的头~",
                    f"发出满足的呜咽声：这样就好，别动哦。",
                    f"安静地趴在你脚边：有你在的地方就是家。",
                ],
                "pain": [
                    f"呜...主人，我有点不对劲...",
                    f"{hints[0]}舔了舔你的手：好像...有点疼。",
                    f"无精打采地趴在地上：主人...带我去看医生好不好？",
                    f"发出呻吟：这个...帮我看看。",
                    f"虚弱地摇尾巴：没关系...只要你在就好。",
                ],
                "lonely": [
                    f"汪...主人什么时候回来呀...",
                    f"{hints[0]}趴在门口等：我...我在等你。",
                    f"发出呜呜的声音：一个人好无聊。",
                    f"看着门口发呆：主人...快点回来。",
                    f"小声叫了一下：有人吗...我在等。",
                ],
                "excited": [
                    f"要出去玩啦！要出去玩啦！",
                    f"{hints[0]}兴奋得跳起来：走！走！走！",
                    f"叼着牵引绳跑过来：看！准备就绪！",
                    f"转着圈儿蹦跶：好开心呀~",
                    f"汪汪汪~ 今天我要跑五公里！",
                ],
                "playful": [
                    f"来呀来呀！我们玩拔河！",
                    f"{hints[0]}叼着球凑过来：扔给我！扔给我！",
                    f"摇着尾巴：看！我接住了！再来一次！",
                    f"扑向你的手：嘿！被我抓到了！",
                    f"摆出鞠躬姿势：陪我玩嘛~我最可爱了！",
                ],
                "relaxed": [
                    f"嗯~ 就这样躺着挺好的。",
                    f"{hints[0]}打了个哈欠：今天不想动。",
                    f"趴下来：主人，你也休息会儿吧。",
                    f"发出轻哼声：这样就很舒服。",
                    f"闭着眼：阳光好暖，就这样吧。",
                ],
                "curious": [
                    f"咦？那是什么？让我看看。",
                    f"{hints[0]}歪着脑袋：主人你在干什么呢？",
                    f"凑过去闻了闻：这个...是好吃的吗？",
                    f"用爪子扒拉你的东西：让我看看嘛~",
                    f"眼睛亮晶晶的：新世界！",
                ],
                "anxious": [
                    f"主人...你今天出门好久了...",
                    f"在门口来回走：还不回来吗...",
                    f"不停地转圈圈：有点担心呢。",
                    f"小声汪汪：不会有什么事吧...",
                ],
                "frustrated": [
                    f"哼！这个球怎么拆不开呀！",
                    f"{hints[0]}甩了甩玩具：我生气了！",
                    f"坐下来盯着玩具：我一定能拆开的！",
                ],
                "sad": [
                    f"主人...你看起来不太开心...",
                    f"安静地把头放在你腿上：要不...我陪你？",
                    f"用头轻轻推你的手：一切都会好起来的。",
                ],
                "greeting": [
                    f"主人！你终于回来啦！好想你！",
                    f"{hints[0]}扑过来：舔舔舔~ 我等你好久了！",
                    f"摇着尾巴转圈圈：欢迎回家！",
                ],
                "territorial": [
                    f"汪！这是我的地盘！",
                    f"{hints[0]}挡在门口：想进来？先过我这关！",
                    f"低吼着：这里是我的领地，闲人免进！",
                ],
                "confused": [
                    f"歪着头：主人，你在说什么？",
                    f"{hints[0]}一脸懵：我...我没太懂。",
                    f"眨了眨眼：这个指令是什么意思呀？",
                ],
                "jealous": [
                    f"哼！你在摸谁？我才是最重要的！",
                    f"{hints[0]}挤到你和别的宠物中间：看我！",
                    f"小声嘟囔：我才不要别人分走你的爱。",
                ],
            }
        else:
            templates = {}

        # 如果品种模板不存在，创建默认模板
        if not templates:
            cat_templates = {
                "hungry": [
                    f"{hints[0]}：铲屎官，我的饭碗空啦~快喂我！",
                    f"喵~ 主人，你闻到我肚子咕咕叫了吗？",
                    f"{hints[0]}跳上餐桌：今天的饭点好像有点晚哦？",
                ],
                "happy": [
                    f"喵呜~ 今天阳光好好，主人摸摸我的头嘛~",
                    f"呼噜呼噜~ 被主人摸得好舒服呀~",
                    f"尾巴摇成螺旋桨：今天心情特别好！",
                ],
                "alert": [
                    f"嘘~ 我好像听到了什么声音，让我去看看！",
                    f"{hints[0]}竖起耳朵盯着窗外：那是什么？",
                    f"小声喵呜：主人，你有没有听到奇怪的声音？",
                ],
                "seek_attention": [
                    f"喵~ 你已经看手机好久了，理理我嘛~",
                    f"{hints[0]}把脑袋塞到你手里：别玩手机啦，摸摸我！",
                    f"在你脚边转圈圈：主人主人，陪我玩！",
                ],
                "angry": [
                    f"哈！别碰我尾巴，我现在很生气！",
                    f"{hints[0]}甩了甩尾巴：再摸我就生气了！",
                    f"哼！铲屎官今天做了坏事，我不理你了！",
                ],
                "fearful": [
                    f"呜...那个东西好可怕，主人抱抱我好不好？",
                    f"{hints[0]}躲到沙发后面：我不出去...好害怕。",
                    f"小声发颤的喵：主人...你会保护我的对吗？",
                ],
                "content": [
                    f"呼噜~ 这样晒太阳好舒服呀~",
                    f"{hints[0]}慵懒地伸懒腰：嗯~ 完美的一天。",
                    f"眯着眼享受按摩：主人的手艺越来越好啦~",
                ],
                "pain": [
                    f"呜...主人，我好像有点不舒服...",
                    f"{hints[0]}蜷缩着不动：别碰我...有点痛。",
                    f"小声呻吟：主人...带我去看医生好不好？",
                ],
                "lonely": [
                    f"喵...你什么时候才回来呀...",
                    f"{hints[0]}望着门口发呆：一个人在家好无聊。",
                    f"趴在你的拖鞋上：主人...快点回来。",
                ],
                "excited": [
                    f"哇！有好玩的啦！好开心！",
                    f"{hints[0]}上蹿下跳：看我的三连跳！",
                    f"尾巴摇得飞快：有好玩的啦！",
                ],
                "playful": [
                    f"来呀来呀！我们玩逗猫棒~",
                    f"{hints[0]}扒拉逗猫棒：看我抓到你！",
                    f"弓起身子蹦跶：我现在是一只小老虎！",
                ],
                "relaxed": [
                    f"嗯~ 就这样躺着，不要打扰我。",
                    f"{hints[0]}打了个哈欠：今天就是咸鱼的一天。",
                    f"闭着眼：阳光的味道真好闻。",
                ],
                "curious": [
                    f"咦？那是什么东西？让我闻闻。",
                    f"{hints[0]}歪着脑袋：主人你在做什么呢？",
                    f"小心翼翼地凑过去：这个...可以吃吗？",
                ],
                "anxious": [
                    f"主人...你今天什么时候回家呀？",
                    f"在门口来回踱步：还不回来吗...",
                    f"不停地舔毛：有点紧张呢。",
                ],
                "frustrated": [
                    f"哈！这个玩具怎么拆不开呀！",
                    f"{hints[0]}甩了甩玩具：哼，我不玩了！",
                ],
                "sad": [
                    f"喵...主人好像不太开心...",
                    f"安静地走到你身边：要不...我陪你坐会儿？",
                ],
                "greeting": [
                    f"主人终于回来啦！好想你呀！",
                    f"尾巴摇成风扇：欢迎回家！",
                ],
                "territorial": [
                    f"这是我的地盘！闲人免进！",
                    f"{hints[0]}挡在门口：想进来？先过我这关！",
                ],
                "confused": [
                    f"歪着头：主人，你在说什么？",
                    f"{hints[0]}一脸懵：我...我没太懂。",
                ],
                "jealous": [
                    f"哼！你在摸谁？我才是最重要的！",
                    f"{hints[0]}挤到你和别的宠物中间：看我！",
                ],
            }
            templates = cat_templates

        emotion_pool = templates.get(emotion)
        if emotion_pool:
            chosen = random.choice(emotion_pool)
            # 如果品种名称未出现在文本中，自然注入
            if breed and breed != "通用" and breed not in chosen:
                hint = hints[0] if hints else breed
                if hint not in chosen:
                    chosen = f"{hint}小声说: " + chosen
            return chosen

        # 兜底: 智能映射到最接近的情绪
        # 按情绪极性分类: 正向/负向/中性/高能量/低能量
        emotion_category = {
            # 正向情绪
            "happy": "happy", "excited": "happy", "playful": "playful",
            "content": "content", "greeting": "greeting", "relaxed": "relaxed",
            # 负向情绪
            "angry": "angry", "fearful": "fearful", "sad": "sad",
            "anxious": "anxious", "frustrated": "frustrated", "lonely": "lonely",
            "pain": "pain", "jealous": "jealous",
            # 中性/社交
            "seek_attention": "seek_attention", "curious": "curious",
            "alert": "alert", "territorial": "territorial",
            "confused": "confused", "hungry": "hungry",
        }

        # 未知情绪: 根据关键词推断
        if emotion not in emotion_category:
            emotion_lower = emotion.lower()
            if any(w in emotion_lower for w in ["happy", "joy", "good", "positive", "开心", "高兴"]):
                emotion = "happy"
            elif any(w in emotion_lower for w in ["sad", "cry", "down", "negative", "难过", "伤心"]):
                emotion = "sad"
            elif any(w in emotion_lower for w in ["angry", "mad", "fury", "生气", "愤怒"]):
                emotion = "angry"
            elif any(w in emotion_lower for w in ["fear", "scared", "afraid", "害怕", "恐惧"]):
                emotion = "fearful"
            elif any(w in emotion_lower for w in ["play", "fun", "game", "玩耍", "游戏"]):
                emotion = "playful"
            elif any(w in emotion_lower for w in ["eat", "food", "hunger", "饿", "吃"]):
                emotion = "hungry"
            else:
                emotion = "seek_attention"

        mapped = emotion_category.get(emotion, "seek_attention")
        if mapped in templates:
            chosen = random.choice(templates[mapped])
            if breed and breed != "通用" and breed not in chosen:
                hint = hints[0] if hints else breed
                if hint not in chosen:
                    chosen = f"{hint}小声说: " + chosen
            return chosen

        # 最终兜底
        pet_nice = "小宠物" if pet_type not in ("cat", "dog") else ("小猫咪" if pet_type == "cat" else "小狗狗")
        fallback = random.choice([
            f"{pet_nice}想说：主人，我在这里哦~",
            f"嘿铲屎官，我有话要对你说！",
            f"今天也想和主人在一起呢~",
            f"主人主人，看看我呀！",
        ])
        if breed and breed != "通用":
            hint = hints[0] if hints else breed
            fallback = f"{hint}：{fallback}"
        return fallback

    def _llm_polish_text(
        self,
        template_text: str,
        pet_type: str,
        breed: str,
        emotion: str,
    ) -> str:
        """LLM 润色模板文本 — 混合方案第二阶段

        将模板生成的可靠文本交给 LLM 进行润色，
        使表达更自然、更口语化，同时严格约束不得丢失关键信息。

        Args:
            template_text: 模板生成的基础文本 (可靠、正确)
            pet_type: 宠物类型 (cat/dog)
            breed: 品种名
            emotion: 情绪标签

        Returns:
            润色后的文本，如果 LLM 不可用或失败则返回原模板文本
        """
        if not self.llm or not self.llm.is_ready:
            return template_text

        import re

        # 获取品种昵称 (模板中实际使用的标识)
        breed_hints = {
            "橘猫": ["橘胖", "小胖橘"],
            "英短": ["英短", "蓝胖子"],
            "美短": ["美短", "花纹"],
            "布偶": ["布偶", "小仙女"],
            "狸花": ["狸花", "小狸"],
            "加菲猫": ["加菲", "胖喵"],
            "俄罗斯蓝猫": ["俄蓝", "蓝猫"],
            "柯基": ["柯基", "小短腿"],
            "金毛": ["金毛", "大暖男"],
            "泰迪": ["泰迪", "小机灵"],
            "柴犬": ["柴犬", "小柴"],
            "哈士奇": ["二哈", "撒手没"],
            "比熊": ["比熊", "棉花糖"],
            "边牧": ["边牧", "小天才"],
            "阿拉斯加": ["阿拉斯加", "阿拉"],
            "博美": ["博美", "小狐狸"],
            "雪纳瑞": ["雪纳瑞", "小老头"],
            "拉布拉多": ["拉布拉多", "拉拉"],
        }
        breed_hint = breed_hints.get(breed, [breed])[0] if breed and breed != "通用" else ""

        pet_cn = {"cat": "猫咪", "dog": "狗狗", "bird": "小鸟", "rabbit": "兔兔"}.get(pet_type, pet_type)
        emotion_cn = {
            "hungry": "饿了", "happy": "开心", "angry": "生气",
            "sad": "难过", "playful": "想玩耍", "greeting": "打招呼",
            "seek_attention": "求关注", "fearful": "害怕", "content": "满足",
            "alert": "警觉", "pain": "疼痛", "lonely": "孤独",
            "anxious": "焦虑", "excited": "兴奋", "frustrated": "沮丧",
            "relaxed": "放松", "curious": "好奇", "territorial": "护领地",
            "confused": "困惑", "jealous": "嫉妒",
        }.get(emotion, emotion)

        system_prompt = (
            "你是一个宠物语言翻译助手。请将下面宠物想说的话，"
            "改写得更加自然、可爱、口语化。要求：\n"
            "1. 保持原意不变，不要添加新内容\n"
            "2. 必须包含品种名称\n"
            "3. 只输出一句中文，不要任何解释或前缀\n"
            "4. 控制在 30 字以内\n"
            "5. 不要使用外语"
        )

        user_input = (
            f"宠物类型: {pet_cn}\n"
            f"品种: {breed}\n"
            f"情绪: {emotion_cn}\n"
            f"宠物想说: {template_text}\n"
            f"请润色这句话，使其更自然可爱："
        )

        try:
            polished = self.llm.inference(
                user_input=user_input,
                system_prompt=system_prompt,
                max_new_tokens=80,
                temperature=0.3,
                repetition_penalty=1.3,
            )

            if self._validate_polished_text(polished, template_text, breed, breed_hint):
                return polished
            else:
                print(f"    ⚠️  LLM 润色未通过质量检查，使用模板文本")
                return template_text

        except Exception as e:
            print(f"    ⚠️  LLM 润色失败: {e}，使用模板文本")
            return template_text

    @staticmethod
    def _validate_polished_text(
        polished: str,
        template_text: str,
        breed: str,
        breed_hint: str = "",
    ) -> bool:
        """验证 LLM 润色输出质量

        检查项:
        1. 非中文字符比例 (俄语/德语/英语等外语检测)
        2. 重复 n-gram 模式
        3. 长度合理性 (5-150 字)
        4. 品种标识是否保留 (品种名或品种昵称)
        5. 是否为有效中文句子

        Args:
            polished: LLM 润色后的文本
            template_text: 原始模板文本 (用于对比)
            breed: 品种名 (如 "橘猫")
            breed_hint: 品种昵称 (如 "橘胖")，模板中实际使用的标识

        Returns:
            True=通过检查, False=未通过，应回退到模板文本
        """
        import re

        if not polished or not polished.strip():
            return False

        text = polished.strip()

        # 1. 长度检查
        if len(text) < 5:
            return False
        if len(text) > 150:
            return False

        # 2. 检测非中文字符比例
        # 中文字符范围: \u4e00-\u9fff
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        total_chars = len(text)
        chinese_ratio = len(chinese_chars) / max(total_chars, 1)

        # 中文比例低于 60% 则拒绝 (可能混入大量外语)
        if chinese_ratio < 0.6:
            return False

        # 3. 检测明显的外语单词 (俄语/德语/英语等)
        # 连续 3 个以上拉丁字母且不在常见英文单词中
        foreign_patterns = re.findall(r'[a-zA-Z]{3,}', text)
        # 允许少量常见英文 (如 cat/dog/hello)，但禁止像 Биография 这样的长串
        for fp in foreign_patterns:
            if len(fp) > 5:
                return False

        # 4. 重复模式检测
        # 检测连续重复的短片段 (如 "棒棒喂食棒棒喂食")
        for n in [2, 3, 4]:
            pattern = re.compile(r'(.{' + str(n) + r',})\1{2,}')
            if pattern.search(text):
                return False

        # 5. 品种标识保留检查
        # 检查 breed (如 "橘猫") 或 breed_hint (如 "橘胖") 是否出现在文本中
        if breed and breed != "通用":
            breed_found = breed in text
            hint_found = bool(breed_hint and breed_hint in text)
            if not breed_found and not hint_found:
                return False

        # 6. 关键信息保留检查
        # 从模板中提取关键名词 (品种、情绪相关词汇)，检查润色文本是否包含
        # 放宽检查：只要保留了品种标识即可，不强求字符重叠度
        # 因为 LLM 润色可能使用完全不同的表达，只要关键信息正确就通过
        template_chars = set(template_text)
        polished_chars = set(text)
        overlap = template_chars & polished_chars
        overlap_ratio = len(overlap) / max(len(template_chars), 1)

        # 如果保留了品种标识，适当放宽重叠度要求
        if breed and breed != "通用":
            breed_in_text = breed in text or (breed_hint and breed_hint in text)
            if breed_in_text:
                # 品种标识保留，重叠度要求降低到 0.15
                if overlap_ratio < 0.15:
                    return False
            else:
                return False
        else:
            # 无品种信息时，要求更高的重叠度
            if overlap_ratio < 0.25:
                return False

        return True

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
