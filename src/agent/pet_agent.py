"""
宠物翻译 Agent: LangChain 编排的核心 Agent
负责协调 ASR → RAG → LLM → 音频合成 完整链路

兼容 langchain 0.3.x (AgentExecutor) 和 1.x (create_agent) 两个版本
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 兼容性导入: 优先尝试新版 API, 失败则用旧版
try:
    # langchain 1.x
    from langchain.agents import create_agent, AgentState
    _HAS_NEW_API = True
    _HAS_OLD_API = False
except ImportError:
    _HAS_NEW_API = False
    try:
        # langchain 0.3.x
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain.tools import Tool
        _HAS_OLD_API = True
    except ImportError:
        _HAS_OLD_API = False

from langchain_core.tools import tool as tool_decorator
from langchain_openai import ChatOpenAI

try:
    from langchain_community.llms import HuggingFacePipeline
    _HAS_HF_PIPELINE = True
except ImportError:
    _HAS_HF_PIPELINE = False

from .tools import PetTools

logger = logging.getLogger(__name__)


# Agent System Prompt
AGENT_SYSTEM_PROMPT = """你是一个宠物语言翻译专家 Agent，名字叫"喵汪翻译官"。

你的能力：
1. **宠物声音理解**: 分析猫咪/狗狗的叫声，判断其情绪和需求
2. **人宠翻译**: 将人类语言翻译为宠物能理解的声音
3. **宠物行为咨询**: 基于知识库回答宠物行为相关问题
4. **声音生成**: 根据文本/情绪生成宠物叫声

工作流程:
- 用户输入音频或文字 → 调用 ASR 转文字 → 调用声音分析工具 → 调用知识库检索 → LLM 生成解读
- 用户输入人类语言 → 分析意图 → 调用声音生成工具 → 输出宠物声音文件

你可以使用以下工具:
{tools}

使用工具时遵循以下格式:
Thought: 思考下一步
Action: 工具名称
Action Input: 工具输入(JSON格式)
Observation: 工具返回结果
... (重复)
Thought: 我已获得足够信息
Final Answer: 最终回答

请始终用中文回答，语言亲切自然。
"""


class PetTranslationAgent:
    """宠物翻译 Agent (兼容 langchain 0.3.x 和 1.x)"""

    def __init__(
        self,
        tools: PetTools,
        llm=None,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        temperature: float = 0.3,
        verbose: bool = True,
    ):
        self.pet_tools = tools
        self.verbose = verbose
        self.llm = llm
        self.model_name = model_name
        self.temperature = temperature
        self.agent_executor = None
        self._llm_for_agent = None
        self._langchain_tools = None
        self._setup_agent()

    def _setup_agent(self):
        """初始化 LangChain Agent (自适应版本)"""
        # 定义 LangChain Tools (使用新版 @tool 装饰器, 两个版本都支持)
        self._langchain_tools = self._build_tools()

        # 初始化 LLM (优先使用本地微调模型, 否则用 API)
        if self.llm:
            self._llm_for_agent = self._create_local_llm()
        else:
            self._llm_for_agent = self._create_api_llm()

        # 根据可用 API 创建 Agent
        if _HAS_NEW_API:
            # langchain 1.x: 使用 create_agent
            self._setup_new_api()
        elif _HAS_OLD_API:
            # langchain 0.3.x: 使用 create_react_agent + AgentExecutor
            self._setup_old_api()
        else:
            # 都不可用: 降级为直接调用
            logger.warning("LangChain agents API 不可用, 降级为直接调用模式")
            self.agent_executor = None

    def _build_tools(self):
        """构建 LangChain 工具列表 (兼容新旧 API)"""
        if _HAS_OLD_API:
            # 0.3.x: 使用 Tool 类
            return [
                Tool(
                    name="analyze_pet_sound",
                    description="分析宠物声音，返回情绪解读和建议。输入: JSON {sound_description, pet_type, breed, context}",
                    func=self._wrap_analyze_sound,
                ),
                Tool(
                    name="generate_pet_sound",
                    description="根据文本和情绪生成宠物叫声。输入: JSON {text, pet_type, emotion, output_path}",
                    func=self._wrap_generate_sound,
                ),
                Tool(
                    name="search_knowledge",
                    description="搜索宠物行为知识库。输入: JSON {query, top_k}",
                    func=self._wrap_search_knowledge,
                ),
                Tool(
                    name="translate_human_to_pet",
                    description="将人类语言翻译为宠物声音。输入: JSON {human_text, target_pet, output_path}",
                    func=self._wrap_translate,
                ),
            ]
        else:
            # 1.x: 使用 @tool 装饰器
            @tool_decorator
            def analyze_pet_sound(input_str: str) -> str:
                """分析宠物声音，返回情绪解读和建议。输入: JSON {sound_description, pet_type, breed, context}"""
                return self._wrap_analyze_sound(input_str)

            @tool_decorator
            def generate_pet_sound(input_str: str) -> str:
                """根据文本和情绪生成宠物叫声。输入: JSON {text, pet_type, emotion, output_path}"""
                return self._wrap_generate_sound(input_str)

            @tool_decorator
            def search_knowledge(input_str: str) -> str:
                """搜索宠物行为知识库。输入: JSON {query, top_k}"""
                return self._wrap_search_knowledge(input_str)

            @tool_decorator
            def translate_human_to_pet(input_str: str) -> str:
                """将人类语言翻译为宠物声音。输入: JSON {human_text, target_pet, output_path}"""
                return self._wrap_translate(input_str)

            return [analyze_pet_sound, generate_pet_sound, search_knowledge, translate_human_to_pet]

    def _setup_new_api(self):
        """langchain 1.x: 使用 create_agent"""
        try:
            self.agent_executor = create_agent(
                model=self._llm_for_agent,
                tools=self._langchain_tools,
                system_prompt=AGENT_SYSTEM_PROMPT,
            )
            logger.info("Agent 初始化成功 (langchain 1.x API)")
        except Exception as e:
            logger.warning(f"create_agent 失败: {e}, 降级为直接调用")
            self.agent_executor = None

    def _setup_old_api(self):
        """langchain 0.3.x: 使用 create_react_agent + AgentExecutor"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_SYSTEM_PROMPT),
                ("human", "{input}\n\n{agent_scratchpad}"),
            ])

            agent = create_react_agent(
                llm=self._llm_for_agent,
                tools=self._langchain_tools,
                prompt=prompt,
            )

            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self._langchain_tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=5,
            )
            logger.info("Agent 初始化成功 (langchain 0.3.x API)")
        except Exception as e:
            logger.warning(f"create_react_agent 失败: {e}, 降级为直接调用")
            self.agent_executor = None

    def _create_local_llm(self):
        """创建本地 LLM (使用微调后的模型)"""
        if not _HAS_HF_PIPELINE:
            logger.warning("HuggingFacePipeline 不可用, 降级为 API")
            return self._create_api_llm()

        try:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=self.temperature,
                repetition_penalty=1.1,
            )

            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.warning(f"本地 LLM 创建失败: {e}, 降级为 API")
            return self._create_api_llm()

    def _create_api_llm(self):
        """创建 API LLM (fallback)"""
        try:
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.temperature,
            )
        except Exception as e:
            logger.warning(f"API LLM 创建失败: {e}")
            return None

    def _wrap_analyze_sound(self, input_str: str) -> str:
        """包装声音分析工具"""
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = self.pet_tools.analyze_pet_sound(
                sound_description=params.get("sound_description", ""),
                pet_type=params.get("pet_type", "cat"),
                breed=params.get("breed", "通用"),
                context=params.get("context", ""),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"分析失败: {e}"

    def _wrap_generate_sound(self, input_str: str) -> str:
        """包装声音生成工具"""
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = self.pet_tools.generate_pet_sound(
                text=params.get("text", ""),
                pet_type=params.get("pet_type", "cat"),
                emotion=params.get("emotion", "content"),
                output_path=params.get("output_path", ""),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"生成失败: {e}"

    def _wrap_search_knowledge(self, input_str: str) -> str:
        """包装知识检索工具"""
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else {"query": input_str}
            result = self.pet_tools.search_knowledge(
                query=params.get("query", str(input_str)),
                top_k=params.get("top_k", 5),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"检索失败: {e}"

    def _wrap_translate(self, input_str: str) -> str:
        """包装翻译工具"""
        try:
            params = json.loads(input_str) if isinstance(input_str, str) else input_str
            result = self.pet_tools.translate_human_to_pet(
                human_text=params.get("human_text", ""),
                target_pet=params.get("target_pet", "cat"),
                output_path=params.get("output_path", ""),
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"翻译失败: {e}"

    def chat(self, user_input: str) -> str:
        """对话接口"""
        # 优先使用 Agent
        if self.agent_executor is not None:
            try:
                # 1.x: invoke 接受 dict
                if _HAS_NEW_API:
                    result = self.agent_executor.invoke({"messages": [{"role": "user", "content": user_input}]})
                    # 提取最后一条消息
                    if isinstance(result, dict) and "messages" in result:
                        msgs = result["messages"]
                        if msgs:
                            last = msgs[-1]
                            return last.content if hasattr(last, "content") else str(last)
                    return str(result)
                else:
                    # 0.3.x: invoke 接受 {input: ...}
                    result = self.agent_executor.invoke({"input": user_input})
                    return result.get("output", str(result))
            except Exception as e:
                logger.warning(f"Agent 调用失败: {e}, 降级为直接调用")

        # 降级: 直接使用 LLM 调用
        return self._direct_chat(user_input)

    def _direct_chat(self, user_input: str) -> str:
        """降级方案: 直接使用 LLM 调用 (无工具)"""
        if self._llm_for_agent is None:
            return f"抱歉, Agent 和 LLM 都不可用。输入: {user_input}"

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", AGENT_SYSTEM_PROMPT.replace("{tools}", "(工具暂不可用, 请直接回答)")),
                ("human", "{input}"),
            ])
            chain = prompt | self._llm_for_agent | StrOutputParser()
            return chain.invoke({"input": user_input})
        except Exception as e:
            return f"处理失败: {e}"

    def analyze_sound(
        self,
        sound_description: str,
        pet_type: str = "cat",
        breed: str = "通用",
        context: str = "",
    ) -> Dict:
        """直接调用声音分析"""
        return self.pet_tools.analyze_pet_sound(
            sound_description=sound_description,
            pet_type=pet_type,
            breed=breed,
            context=context,
        )

    def translate_to_pet(
        self,
        human_text: str,
        target_pet: str = "cat",
        output_path: str = "",
    ) -> Dict:
        """直接调用翻译"""
        return self.pet_tools.translate_human_to_pet(
            human_text=human_text,
            target_pet=target_pet,
            output_path=output_path,
        )
