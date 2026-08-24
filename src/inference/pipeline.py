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
        print(f"  ┌─ 模块 3/3: Bark 音频生成 (宠物声音合成)")
        try:
            self.audio_generator = AudioGenerator()
            if self.audio_generator.is_ready:
                print(f"  └─ ✅ Bark 模型加载成功")
            else:
                print(f"  └─ ⚠️  Bark 不可用，降级为模拟音频生成")
        except Exception as e:
            print(f"  └─ ⚠️  音频生成器加载失败: {e}")
            self.audio_generator = None

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
        """宠物声音 → 人类语言 — 按业务架构分步打印"""
        result = {
            "input_audio": audio_path,
            "pet_type": pet_type,
            "rag_details": [],
        }

        # ============ LangChain Agent 编排层 Step 1: 声音分析 ============
        print(f"\n{'='*60}")
        print(f"🕹️ 【LangChain Agent 编排层】执行: 宠物声音 → 人类语言")
        print(f"{'='*60}")
        print(f"  Step 1/3: 声音分析 Tool")
        print(f"    ↓ 输入: 音频文件 → Whisper ASR (模型服务层)")

        # 1. Whisper ASR (模型服务层)
        if self.whisper:
            whisper_result = self.whisper.inference(audio_path)
            result["whisper_transcription"] = whisper_result
        else:
            whisper_result = "无法识别(Whisper 未加载)"
            result["whisper_transcription"] = whisper_result
        print(f"    ↑ 输出: '{whisper_result}'")

        # ============ LangChain Agent 编排层 Step 2: 知识检索 ============
        print(f"  Step 2/3: 知识检索 Tool")
        print(f"    ↓ 输入: '{whisper_result}' → Embedding + ChromaDB (RAG 知识检索层)")

        # 2. RAG 检索 (RAG 知识检索层)
        if self.rag_retriever and self.rag_retriever.is_ready:
            search_detail = self.rag_retriever.search_with_details(
                f"{pet_type} {whisper_result} 情绪 含义 建议"
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
        print(f"  Step 3/3: 情绪解读 Tool")
        print(f"    ↓ 输入: Whisper输出 + RAG上下文 → LLM LoRA (模型服务层)")

        # 3. LLM 分析
        if self.agent:
            analysis = self.agent.analyze_sound(
                sound_description=whisper_result,
                pet_type=pet_type,
                breed=breed,
            )
            result["analysis"] = analysis.get("analysis", str(analysis))
        elif self.llm:
            rag_context = ""
            if self.rag_retriever and self.rag_retriever.is_ready:
                rag_context = self.rag_retriever.get_context_for_llm(
                    f"{pet_type} {whisper_result}"
                )

            system_prompt = "你是一个宠物行为学专家。"
            user_input = (
                f"宠物类型: {pet_type}, 品种: {breed}\n"
                f"声音描述: {whisper_result}\n\n"
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
