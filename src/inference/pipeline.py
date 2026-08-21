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
        use_rag: bool = True,
        use_agent: bool = True,
    ):
        """
        Args:
            whisper_model_path: Whisper 微调模型路径
            llm_adapter_path: LLM LoRA adapter 路径
            use_rag: 是否启用 RAG
            use_agent: 是否使用 Agent 编排
        """
        self.audio_processor = AudioProcessor()
        self.whisper = None
        self.llm = None
        self.rag_retriever = None
        self.audio_generator = None
        self.agent = None

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
        """初始化各组件 (网络断开时优雅降级)"""
        # 1. Whisper ASR
        whisper_config = WhisperFineTuneConfig()
        if whisper_path and os.path.exists(whisper_path):
            try:
                self.whisper = WhisperFineTuner(whisper_config)
                self.whisper.load_finetuned(whisper_path)
            except Exception as e:
                print(f"⚠️  Whisper 微调模型加载失败: {e}")
                self.whisper = None
        else:
            try:
                self.whisper = WhisperFineTuner(whisper_config)
                if not self.whisper.is_ready:
                    print("⚠️  Whisper 基础模型未就绪 (网络不可用或未缓存)")
                    print("   pet_to_text 模式需要 Whisper，可尝试:")
                    print("     1. 连接网络后重试")
                    print("     2. 使用已训练好的 Whisper 模型: --whisper_model ./checkpoints/whisper/final")
                    print("     3. 下载 whisper-tiny 模型到本地缓存")
                    print("   text_to_pet / chat 模式仍可正常使用")
                    self.whisper = None
            except Exception as e:
                print(f"⚠️  Whisper 加载失败: {e}")
                self.whisper = None

        # 2. LLM
        llm_config = LLMFineTuneConfig()
        if llm_path and os.path.exists(llm_path):
            try:
                self.llm = LLMFineTuner(llm_config)
                self.llm.load_finetuned(llm_path)
            except Exception as e:
                print(f"⚠️  LLM adapter 加载失败: {e}")
                self.llm = None
        else:
            try:
                self.llm = LLMFineTuner(llm_config)
                if not self.llm.is_ready:
                    print("⚠️  LLM 基础模型未就绪 (网络不可用或未缓存)")
                    self.llm = None
            except Exception as e:
                print(f"⚠️  LLM 加载失败: {e}")
                self.llm = None

        # 3. RAG
        if use_rag:
            kb = PetKnowledgeBase()
            self.rag_retriever = PetRetriever(knowledge_base=kb)
            try:
                self.rag_retriever.build_index()
            except Exception as e:
                print(f"⚠️  RAG 索引构建失败: {e}")

        # 4. 音频生成
        try:
            self.audio_generator = AudioGenerator()
        except Exception as e:
            print(f"⚠️  音频生成器加载失败: {e}")
            self.audio_generator = None

        # 5. Agent
        if use_agent and self.llm:
            tools = PetTools(
                retriever=self.rag_retriever,
                audio_generator=self.audio_generator,
                llm=self.llm,
            )
            try:
                self.agent = PetTranslationAgent(tools=tools, llm=self.llm)
            except Exception as e:
                print(f"⚠️  Agent 初始化失败(降级为直接调用): {e}")
                self.agent = None
        else:
            self.agent = None

    def pet_sound_to_text(
        self,
        audio_path: str,
        pet_type: str = "cat",
        breed: str = "通用",
    ) -> Dict:
        """宠物声音 → 人类语言"""
        result = {
            "input_audio": audio_path,
            "pet_type": pet_type,
        }

        # 1. Whisper ASR
        if self.whisper:
            whisper_result = self.whisper.inference(audio_path)
            result["whisper_transcription"] = whisper_result
        else:
            whisper_result = "无法识别(Whisper 未加载)"
            result["whisper_transcription"] = whisper_result

        # 2. RAG + LLM 分析
        if self.agent:
            analysis = self.agent.analyze_sound(
                sound_description=whisper_result,
                pet_type=pet_type,
                breed=breed,
            )
            result["analysis"] = analysis.get("analysis", str(analysis))
        elif self.llm and self.rag_retriever:
            context = self.rag_retriever.get_context_for_llm(whisper_result)
            system_prompt = "你是一个宠物行为学专家。"
            user_input = (
                f"宠物类型: {pet_type}, 品种: {breed}\n"
                f"声音描述: {whisper_result}\n\n"
                f"知识库参考:\n{context}\n"
                f"请分析情绪和需求。"
            )
            result["analysis"] = self.llm.inference(user_input, system_prompt)
        else:
            result["analysis"] = "LLM 未加载"

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
            audio = self.audio_generator.human_speech_to_pet_sound(
                speech_text=text,
                target_pet=target_pet,
                output_path=output_path if output_path else None,
            )
            result["status"] = "success"
            result["output_path"] = output_path
        else:
            result["status"] = "no_audio_generator"

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
