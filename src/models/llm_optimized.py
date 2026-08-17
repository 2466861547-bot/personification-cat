"""
LLM 优化推理器: 基于 vLLM 的高性能推理
- PagedAttention: 3-5x 加速
- AWQ 4bit 量化: 显存减半
- Speculative Decoding: draft model 加速
- Continuous Batching: 批量推理优化
"""

import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass

try:
    from vllm import LLM as VLLMModel, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@dataclass
class OptimizedLLMConfig:
    """vLLM 优化配置"""
    model_path: str = "./checkpoints/llm/final"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # vLLM 参数
    tensor_parallel_size: int = 1        # GPU 并行数
    gpu_memory_utilization: float = 0.9  # GPU 显存利用率
    max_model_len: int = 4096            # 最大序列长度
    dtype: str = "bfloat16"              # bfloat16 / float16 / auto
    quantization: Optional[str] = None   # awq / gptq / None
    # Speculative Decoding
    draft_model: Optional[str] = None    # draft model 路径
    num_speculative_tokens: int = 5      # 投机解码 token 数
    # 生成参数
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 512
    # KV Cache
    enable_prefix_caching: bool = True   # 前缀缓存(相同 system prompt 加速)
    enable_chunked_prefill: bool = True  # 分块预填充


class OptimizedLLMInference:
    """
    vLLM 优化推理器
    加速: PagedAttention + Prefix Caching + Continuous Batching
    量化: AWQ 4bit (显存 24GB → 12GB)
    投机解码: draft model 预测 + verify
    """

    def __init__(self, config: OptimizedLLMConfig = None):
        self.config = config or OptimizedLLMConfig()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载 vLLM 模型"""
        if not VLLM_AVAILABLE:
            print("vLLM 未安装,请安装: pip install vllm")
            return

        model_path = self.config.model_path
        if not os.path.exists(model_path):
            model_path = self.config.base_model
            print(f"微调模型不存在,使用基础模型: {model_path}")

        print(f"加载 vLLM 模型: {model_path}")
        print(f"  量化: {self.config.quantization or 'none'}")
        print(f"  Draft model: {self.config.draft_model or 'none'}")

        kwargs = {
            "model": model_path,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "dtype": self.config.dtype,
            "trust_remote_code": True,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
        }

        # 量化
        if self.config.quantization:
            kwargs["quantization"] = self.config.quantization

        # 投机解码
        if self.config.draft_model:
            kwargs["speculative_model"] = self.config.draft_model
            kwargs["num_speculative_tokens"] = self.config.num_speculative_tokens

        self.model = VLLMModel(**kwargs)

        # 加载 tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
    ) -> str:
        """单条生成"""
        if self.model is None:
            return "vLLM 未加载"

        # 构建 prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        sampling_params = SamplingParams(
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = None,
        max_tokens: int = None,
    ) -> List[str]:
        """批量生成 (Continuous Batching)"""
        if self.model is None:
            return ["vLLM 未加载"] * len(messages_list)

        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_list
        ]

        sampling_params = SamplingParams(
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens or self.config.max_tokens,
        )

        outputs = self.model.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]

    def chat(
        self,
        user_input: str,
        system_prompt: str = "你是一个宠物语言翻译专家。",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """对话接口"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        return self.generate(messages, temperature, max_tokens)

    def analyze_pet_emotion(
        self,
        sound_description: str,
        pet_type: str = "cat",
        breed: str = "通用",
        context: str = "",
        knowledge_context: str = "",
    ) -> str:
        """宠物情绪分析 (专用接口)"""
        system_prompt = (
            "你是一个宠物行为学专家，擅长根据宠物声音判断情绪。"
            "请给出简洁专业的分析，包含情绪、含义和建议。"
        )
        user_input = (
            f"宠物: {pet_type}({breed})\n"
            f"声音: {sound_description}\n"
            f"情境: {context}\n"
        )
        if knowledge_context:
            user_input += f"\n知识库:\n{knowledge_context}\n"
        user_input += "请分析情绪和需求。"

        return self.chat(user_input, system_prompt, max_tokens=256)


class ModelQuantizer:
    """
    模型量化工具: 将微调后的模型量化为 AWQ/GPTQ 格式
    减少显存占用 50%+, 推理速度提升 1.5-2x
    """

    @staticmethod
    def quantize_awq(
        model_path: str,
        output_path: str,
        calibration_data: List[str] = None,
    ):
        """AWQ 量化"""
        print(f"AWQ 量化: {model_path} → {output_path}")

        try:
            from awq import AutoAWQForCausalLM
            from transformers import AutoTokenizer

            model = AutoAWQForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            # 默认校准数据
            if calibration_data is None:
                calibration_data = [
                    "宠物类型: cat，品种: 橘猫\n声音: 短促喵叫\n情境: 饭点",
                    "宠物类型: dog，品种: 金毛\n声音: 摇尾巴汪汪叫\n情境: 主人回家",
                    "宠物类型: cat，品种: 布偶\n声音: 呼噜声\n情境: 被抚摸",
                ]

            quant_config = {
                "zero_point": True,
                "q_group_size": 128,
                "w_bit": 4,
                "version": "GEMM",
            }

            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data,
            )

            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"AWQ 量化完成: {output_path}")

        except ImportError:
            print("awq 未安装,请安装: pip install autoawq")

    @staticmethod
    def merge_lora_and_quantize(
        base_model: str,
        lora_adapter: str,
        output_path: str,
        quantization: str = "awq",
    ):
        """合并 LoRA adapter 后量化"""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"合并 LoRA: {base_model} + {lora_adapter}")

        # 1. 加载基础模型
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )

        # 2. 加载并合并 LoRA
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()

        # 3. 保存合并后的模型
        merged_path = output_path + "_merged"
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"合并模型已保存: {merged_path}")

        # 4. 量化
        if quantization == "awq":
            ModelQuantizer.quantize_awq(merged_path, output_path)
