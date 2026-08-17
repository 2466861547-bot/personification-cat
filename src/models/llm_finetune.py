"""
LLM LoRA 微调模块: 微调大语言模型实现宠物声音理解 + 人宠翻译
任务:
  1. 宠物声音描述 → 情绪/意图解读
  2. 人类语言 → 宠物声音生成指令
"""

import os
import json
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset


@dataclass
class LLMFineTuneConfig:
    """LLM 微调配置"""
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = None
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 10
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    save_steps: int = 200
    eval_steps: int = 200
    bf16: bool = True
    output_dir: str = "./checkpoints/llm"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]


class SFTDataset(torch.utils.data.Dataset):
    """监督微调数据集"""

    def __init__(
        self,
        data_list: List[Dict],
        tokenizer,
        max_length: int = 2048,
    ):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建对话格式
        system_prompt = item.get("system", "你是一个宠物语言翻译专家。")
        user_input = item["input"]
        response = item["output"]

        # 使用 chat template 格式化
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # 分词
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 构建标签: 只对 assistant 部分计算 loss
        labels = input_ids.clone()

        # 找到 assistant 部分的起始位置
        assistant_start = text.find("<|im_start|>assistant")
        if assistant_start == -1:
            assistant_start = text.find("assistant")

        # 对 system 和 user 部分标记为 -100 (不计算 loss)
        prefix_text = text[:assistant_start]
        prefix_encoding = self.tokenizer(
            prefix_text, truncation=True, max_length=self.max_length
        )
        prefix_len = len(prefix_encoding["input_ids"])

        labels[:prefix_len] = -100
        # padding 部分也标记为 -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LLMFineTuner:
    """LLM LoRA 微调器"""

    def __init__(self, config: LLMFineTuneConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载基础模型"""
        print(f"加载基础模型: {self.config.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.config.target_modules,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
    ) -> tuple:
        """准备训练数据集"""
        train_dataset = SFTDataset(
            data_list=train_data,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length,
        )

        eval_dataset = None
        if eval_data:
            eval_dataset = SFTDataset(
                data_list=eval_data,
                tokenizer=self.tokenizer,
                max_length=self.config.max_seq_length,
            )

        return train_dataset, eval_dataset

    def train(self, train_data: List[Dict], eval_data: Optional[List[Dict]] = None):
        """启动 LoRA 微调训练"""
        train_dataset, eval_dataset = self.prepare_dataset(train_data, eval_data)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_total_limit=3,
            bf16=self.config.bf16,
            logging_steps=20,
            report_to="tensorboard",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            gradient_checkpointing=True,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print("开始 LLM LoRA 微调训练...")
        trainer.train()

        # 保存 LoRA adapter
        save_path = os.path.join(self.config.output_dir, "final")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"LoRA adapter 已保存到: {save_path}")

        return save_path

    def inference(
        self,
        user_input: str,
        system_prompt: str = "你是一个宠物语言翻译专家。",
        max_new_tokens: int = 512,
    ) -> str:
        """推理"""
        self.model.eval()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        response = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def load_finetuned(self, adapter_path: str):
        """加载已微调的 LoRA adapter"""
        from peft import PeftModel

        # 先加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        # 加载 LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"已加载微调模型: {adapter_path}")
