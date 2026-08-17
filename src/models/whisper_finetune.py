"""
Whisper 微调模块: 将 Whisper 微调为宠物声音识别器
输入: 宠物叫声音频 → 输出: 文字描述(情绪 + 意图)
"""

import os
import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class WhisperFineTuneConfig:
    """Whisper 微调配置"""
    model_name: str = "openai/whisper-large-v3"
    language: str = "zh"
    task: str = "transcribe"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 1e-5
    batch_size: int = 8
    num_train_epochs: int = 30
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 500
    fp16: bool = True
    gradient_checkpointing: bool = True
    output_dir: str = "./checkpoints/whisper"


class PetSoundDataset(Dataset):
    """宠物声音训练数据集"""

    def __init__(
        self,
        data_list: list,
        processor: WhisperProcessor,
        max_duration: float = 5.0,
        sample_rate: int = 16000,
    ):
        """
        Args:
            data_list: [{"audio_path": ..., "text": ...}]
            processor: WhisperProcessor
            max_duration: 最大音频时长
        """
        self.data_list = data_list
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        import librosa

        item = self.data_list[idx]
        audio_path = item["audio_path"]
        text = item["text"]

        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # 截断/填充
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        else:
            audio = np.pad(audio, (0, self.max_samples - len(audio)))

        # 处理为 Whisper 输入
        processed = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            text=text,
            return_tensors="pt",
        )

        # 去除 batch 维度
        input_features = processed.input_features.squeeze(0)
        labels = processed.labels.squeeze(0)

        return {
            "input_features": input_features,
            "labels": labels,
        }


class WhisperFineTuner:
    """Whisper 模型微调器"""

    def __init__(self, config: WhisperFineTuneConfig):
        self.config = config
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载预训练模型和处理器"""
        print(f"加载 Whisper 模型: {self.config.model_name}")

        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )

        if self.config.gradient_checkpointing:
            self.model.config.use_cache = False
            self.model.gradient_checkpointing_enable()

        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules=["q_proj", "v_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        train_data: list,
        eval_data: Optional[list] = None,
    ) -> tuple:
        """准备训练和验证数据集"""
        train_dataset = PetSoundDataset(
            data_list=train_data,
            processor=self.processor,
            max_duration=5.0,
        )

        eval_dataset = None
        if eval_data:
            eval_dataset = PetSoundDataset(
                data_list=eval_data,
                processor=self.processor,
                max_duration=5.0,
            )

        return train_dataset, eval_dataset

    def train(self, train_data: list, eval_data: Optional[list] = None):
        """启动微调训练"""
        train_dataset, eval_dataset = self.prepare_dataset(train_data, eval_data)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_total_limit=3,
            fp16=self.config.fp16,
            logging_steps=50,
            report_to="tensorboard",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )

        # data collator
        def data_collator(features):
            input_features = torch.stack([f["input_features"] for f in features])
            label_features = [f["labels"] for f in features]
            labels_batch = self.processor.label_processor.pad(
                {"input_ids": label_features}, return_tensors="pt"
            )
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            return {"input_features": input_features, "labels": labels}

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.processor.tokenizer,
        )

        print("开始 Whisper 微调训练...")
        trainer.train()

        # 保存最终模型
        save_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

        return save_path

    def inference(self, audio_path: str) -> str:
        """推理: 宠物声音 → 文字描述"""
        import librosa

        self.model.eval()
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        processed = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = processed.input_features

        if torch.cuda.is_available():
            input_features = input_features.cuda()
            self.model = self.model.cuda()

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=200,
                language=self.config.language,
                task=self.config.task,
            )

        text = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return text

    def load_finetuned(self, model_path: str):
        """加载已微调的模型"""
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print(f"已加载微调模型: {model_path}")
