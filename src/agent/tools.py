"""
Agent 工具集: 定义 LangChain Agent 可调用的工具
"""

import os
from typing import Dict, Optional
from langchain_core.tools import tool

# 兼容性导入: langchain 0.3.x 用 pydantic_v1, 1.x 直接用 pydantic
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except (ImportError, ModuleNotFoundError):
    from pydantic import BaseModel, Field


class AnalyzeSoundInput(BaseModel):
    """声音分析输入"""
    sound_description: str = Field(description="宠物声音的文字描述")
    pet_type: str = Field(description="宠物类型: cat 或 dog")
    breed: str = Field(default="通用", description="宠物品种")
    context: str = Field(default="", description="情境描述")


class GeneratePetSoundInput(BaseModel):
    """宠物声音生成输入"""
    text: str = Field(description="要翻译的文本")
    pet_type: str = Field(description="目标宠物类型: cat 或 dog")
    emotion: str = Field(default="content", description="情绪标签")
    output_path: str = Field(default="", description="输出文件路径")


class PetTools:
    """宠物翻译 Agent 工具集"""

    def __init__(self, retriever=None, audio_generator=None, llm=None):
        self.retriever = retriever
        self.audio_generator = audio_generator
        self.llm = llm

    def analyze_pet_sound(
        self,
        sound_description: str,
        pet_type: str,
        breed: str = "通用",
        context: str = "",
    ) -> Dict:
        """分析宠物声音,返回情绪解读"""
        # 1. RAG 检索相关知识
        knowledge_context = ""
        if self.retriever:
            query = f"{pet_type} {sound_description} 情绪 含义"
            knowledge_context = self.retriever.get_context_for_llm(query)

        # 2. LLM 分析
        if self.llm:
            system_prompt = (
                "你是一个宠物行为学专家，擅长根据宠物的声音特征判断其情绪和需求。"
                "请结合知识库内容，给出专业的分析。"
            )
            user_input = (
                f"宠物类型: {pet_type}\n"
                f"品种: {breed}\n"
                f"声音特征: {sound_description}\n"
                f"情境: {context}\n\n"
                f"【知识库参考】\n{knowledge_context}\n\n"
                f"请分析这只{pet_type}的情绪状态、含义和推荐措施。"
            )
            analysis = self.llm.inference(user_input, system_prompt)
        else:
            analysis = f"[需加载 LLM 模型] 宠物({pet_type})声音分析: {sound_description}"

        return {
            "pet_type": pet_type,
            "breed": breed,
            "sound_description": sound_description,
            "context": context,
            "analysis": analysis,
            "knowledge_used": bool(knowledge_context),
        }

    def generate_pet_sound(
        self,
        text: str,
        pet_type: str,
        emotion: str = "content",
        output_path: str = "",
    ) -> Dict:
        """生成宠物声音"""
        if self.audio_generator:
            audio = self.audio_generator.text_to_pet_sound(
                text=text,
                pet_type=pet_type,
                emotion=emotion,
                output_path=output_path if output_path else None,
            )
            return {
                "text": text,
                "pet_type": pet_type,
                "emotion": emotion,
                "output_path": output_path,
                "audio_length": len(audio),
                "status": "success",
            }
        else:
            return {
                "text": text,
                "pet_type": pet_type,
                "emotion": emotion,
                "status": "no_audio_generator",
            }

    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """搜索宠物知识库"""
        if self.retriever:
            results = self.retriever.search(query, top_k=top_k)
            return {
                "query": query,
                "results_count": len(results),
                "results": results,
            }
        return {"query": query, "results": [], "status": "no_retriever"}

    def translate_human_to_pet(
        self,
        human_text: str,
        target_pet: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """人类语言 → 宠物声音"""
        if self.audio_generator:
            audio = self.audio_generator.human_speech_to_pet_sound(
                speech_text=human_text,
                target_pet=target_pet,
                output_path=output_path if output_path else None,
            )
            return {
                "human_text": human_text,
                "target_pet": target_pet,
                "output_path": output_path,
                "status": "success",
            }
        return {
            "human_text": human_text,
            "target_pet": target_pet,
            "status": "no_audio_generator",
        }
