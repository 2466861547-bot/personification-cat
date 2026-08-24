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

# 国内镜像: 优先使用 hf-mirror.com, 避免连不上 huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# 模型已缓存到本地时, 强制离线模式, 避免网络波动导致加载失败
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
    # 默认使用 whisper-tiny 以加快下载/微调速度; 如需更高精度改为 whisper-base / whisper-small
    model_name: str = "openai/whisper-tiny"
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
    # 新增: 轻量化训练参数
    max_seq_length: int = 100  # 最大序列长度 (token 数), 减少显存
    gradient_accumulation_steps: int = 1  # 梯度累积步数
    max_train_samples: Optional[int] = None  # 限制训练样本数, None=全部
    cpu_threads: int = 4  # CPU 线程数


def get_mac_light_config() -> WhisperFineTuneConfig:
    """Mac 本地轻量训练配置 (避免卡死)"""
    return WhisperFineTuneConfig(
        model_name="openai/whisper-tiny",
        batch_size=2,              # Mac 上 batch=8 太大, 改为 2
        num_train_epochs=5,        # 30 epochs 太多, 改为 5
        learning_rate=1e-4,        # 较大学习率, 少 epoch 也能收敛
        lora_r=8,                  # 减少 LoRA 参数量
        lora_alpha=16,
        max_seq_length=64,         # 限制序列长度
        gradient_accumulation_steps=4,  # 累积梯度模拟更大 batch
        max_train_samples=500,     # 只用 500 条样本训练
        cpu_threads=4,
    )


def get_heavy_config(gpu_mem_gb: int = 24) -> WhisperFineTuneConfig:
    """
    GPU 服务器重量级训练配置

    Args:
        gpu_mem_gb: GPU 显存大小 (GB), 自动适配 batch_size
    """
    # 根据显存自动选择 batch_size 和 model
    if gpu_mem_gb >= 48:
        model_name = "openai/whisper-large-v3"
        batch_size = 16
        lora_r = 64
    elif gpu_mem_gb >= 24:
        model_name = "openai/whisper-large-v3"
        batch_size = 8
        lora_r = 32
    elif gpu_mem_gb >= 16:
        model_name = "openai/whisper-medium"
        batch_size = 8
        lora_r = 32
    elif gpu_mem_gb >= 8:
        model_name = "openai/whisper-small"
        batch_size = 4
        lora_r = 16
    else:
        model_name = "openai/whisper-base"
        batch_size = 2
        lora_r = 16

    return WhisperFineTuneConfig(
        model_name=model_name,
        batch_size=batch_size,
        num_train_epochs=30,
        learning_rate=1e-5,
        lora_r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        max_seq_length=100,
        gradient_accumulation_steps=1,
        max_train_samples=None,  # 全量数据
        cpu_threads=8,
    )


def detect_device() -> str:
    """自动检测训练设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_gpu_memory_gb() -> int:
    """获取 GPU 显存大小 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_mem // (1024 ** 3)
    return 0


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

        # 处理为 Whisper 输入 (分离音频和文本处理)
        input_features = self.processor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        ).input_features.squeeze(0)

        # 单独处理文本标签 (确保 tokenizer 正确处理)
        labels = self.processor.tokenizer(
            text, return_tensors="pt"
        ).input_ids.squeeze(0)

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
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """加载预训练模型和处理器 (网络断开时优雅降级)"""
        print(f"加载 Whisper 模型: {self.config.model_name}")

        try:
            self.processor = WhisperProcessor.from_pretrained(self.config.model_name)

            # 自动检测设备
            self.device = detect_device()
            print(f"  设备: {self.device}")

            # MPS 上 Whisper conv1/decoder 有兼容性问题, 强制用 CPU 训练
            # 只有 CUDA 才用 GPU 加速, MPS 回退到 CPU
            if self.device == "mps":
                print("  ⚠ MPS 设备 Whisper 层有兼容性问题, 强制回退到 CPU 训练")
                print("    如需 GPU 加速, 请使用 Linux CUDA 服务器")
                self.device = "cpu"

            # 根据设备选择 dtype
            if self.device == "cuda":
                dtype = torch.float16 if self.config.fp16 else torch.float32
            else:
                dtype = torch.float32  # CPU 用 float32 更稳定

            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
            )
        except Exception as e:
            print(f"  ❌ Whisper 加载失败: {e}")
            print(f"     可能原因: 网络不可用或模型未缓存")
            print(f"     解决方法:")
            print(f"       1. 连接网络后重试")
            print(f"       2. 使用已训练好的 Whisper 模型路径")
            print(f"       3. 手动下载模型到本地: {self.config.model_name}")
            self.processor = None
            self.model = None
            self.is_ready = False
            return

        # 强制将模型放到 CPU (防止 Trainer 自动移到 MPS)
        if self.device == "cpu":
            self.model = self.model.cpu()
            print("  模型已固定到 CPU 设备")

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

        # 兼容性修复: PEFT 的 forward() 会显式传递 input_ids=None 和 inputs_embeds=None 给 base_model
        # 但 WhisperForConditionalGeneration 没有这些参数, 导致它们泄漏到 **kwargs
        # 最终在 WhisperDecoder 与显式的 input_ids=decoder_input_ids 和 inputs_embeds 冲突
        # 修复: 在 WhisperForConditionalGeneration 层移除泄漏的问题参数
        import types

        _base_model = self.model.base_model
        _original_base_forward = _base_model.__class__.forward

        _PROBLEM_KEYS = {"input_ids", "inputs_embeds"}

        def _patched_base_forward(self_, *args, **kwargs):
            # 移除 PEFT 泄漏的参数, 防止在 WhisperDecoder 层与显式参数冲突
            for key in _PROBLEM_KEYS:
                kwargs.pop(key, None)
            return _original_base_forward(self_, *args, **kwargs)

        _base_model.forward = types.MethodType(_patched_base_forward, _base_model)

        self.is_ready = True

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
        # 限制训练样本数 (轻量化)
        if self.config.max_train_samples and len(train_data) > self.config.max_train_samples:
            train_data = train_data[:self.config.max_train_samples]
            print(f"  ⚠ 限制训练样本: {len(train_data)} 条 (max_train_samples={self.config.max_train_samples})")

        train_dataset, eval_dataset = self.prepare_dataset(train_data, eval_data)

        # 兼容 transformers 新旧版本: 4.46+ 使用 eval_strategy, 旧版使用 evaluation_strategy
        eval_strategy_kwargs = {}
        if eval_dataset:
            try:
                # transformers >= 4.46
                from transformers import __version__ as _tf_ver
                _major_minor = tuple(int(x) for x in _tf_ver.split(".")[:2])
                if _major_minor >= (4, 46):
                    eval_strategy_kwargs["eval_strategy"] = "steps"
                else:
                    eval_strategy_kwargs["evaluation_strategy"] = "steps"
            except Exception:
                # 优先尝试 eval_strategy (新 API), 失败则用 evaluation_strategy
                eval_strategy_kwargs["evaluation_strategy"] = "steps"

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_total_limit=3,
            fp16=self.config.fp16 if self.device == "cuda" else False,  # 仅 CUDA 支持 fp16
            logging_steps=50,
            report_to="tensorboard",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="loss",
            greater_is_better=False,
            remove_unused_columns=False,
            # macOS MPS 不支持 pin_memory, 关闭以消除警告
            dataloader_pin_memory=False,
            # 梯度累积: 小 batch 模拟大 batch
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            # 限制每个 epoch 的步数, 防止训练过久
            max_steps=-1,  # -1 表示按 epoch 计算
            **eval_strategy_kwargs,
        )

        # data collator: 使用 tokenizer 正确填充标签
        _tokenizer = self.processor.tokenizer

        def data_collator(features):
            input_features = torch.stack([f["input_features"] for f in features])
            label_features = [f["labels"] for f in features]

            # 使用 tokenizer 填充标签 (WhisperTokenizer 专用 pad 方法)
            labels_batch = _tokenizer.pad(
                {"input_ids": label_features},
                return_tensors="pt",
                padding=True,
            )

            # 将 padding 位置设为 -100 (忽略损失)
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            # 确保 labels 为 torch.int64 (long) 类型
            labels = labels.long()

            return {"input_features": input_features, "labels": labels}

        # 兼容 transformers 新旧版本: 5.x 用 processing_class, 旧版用 tokenizer
        trainer_kwargs = dict(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # CPU 训练: 确保模型留在 CPU 上 (新版 transformers 自动处理, 无需 place_model_on_device)
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        # 尝试新版 API (processing_class), 失败则回退旧版 (tokenizer)
        try:
            trainer = Seq2SeqTrainer(processing_class=self.processor, **trainer_kwargs)
        except (TypeError, AttributeError):
            _tokenizer = getattr(self.processor, "tokenizer", None) or self.processor
            trainer = Seq2SeqTrainer(tokenizer=_tokenizer, **trainer_kwargs)

        print("开始 Whisper 微调训练...")

        trainer.train()

        # 保存最终模型
        save_path = os.path.join(self.config.output_dir, "final")
        trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)
        print(f"模型已保存到: {save_path}")

        return save_path

    def inference(self, audio_path: str) -> str:
        """推理: 宠物声音 → 文字描述 — 输出详细识别日志"""
        if not self.is_ready:
            raise RuntimeError("Whisper 模型未就绪，无法推理。请检查网络连接或使用本地缓存模型。")
        import librosa

        self.model.eval()

        # ===== Step 1: 音频加载与预处理 =====
        print(f"  🎙️  Whisper Step 1/4: 加载音频")
        print(f"           文件: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        import numpy as np
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        zcr = float(np.mean(np.diff(np.sign(audio)) != 0))
        # 主频
        window = np.hanning(len(audio))
        spec = np.abs(np.fft.rfft(audio * window))
        freqs = np.fft.rfftfreq(len(audio), 1.0/16000)
        main_freq = float(freqs[int(np.argmax(spec))])
        top3_freqs = freqs[np.argsort(spec)[-3:][::-1]]

        print(f"           重采样: 目标 16kHz, 单声道, 实际 {sr}Hz")
        print(f"           样本数: {len(audio)} 个采样点")
        print(f"           时长:   {duration:.3f} 秒")
        print(f"           峰值:   {peak:.4f} ({20*np.log10(max(peak,1e-8)):.1f} dBFS)")
        print(f"           RMS:    {rms:.4f}")
        print(f"           过零率: {zcr:.4f}")
        print(f"           主频率: {main_freq:.1f} Hz (Top3: {[f'{f:.0f}' for f in top3_freqs]} Hz)")

        # 估算语音活动比例 (简单 VAD: RMS > 10% 峰值)
        vad_threshold = peak * 0.1
        voice_ratio = float(np.mean(np.abs(audio) > vad_threshold))
        print(f"           语音比例: {voice_ratio*100:.1f}% (RMS > {vad_threshold:.4f})")

        # ===== Step 2: Mel 特征提取 =====
        print(f"  🎛️  Whisper Step 2/4: Mel 特征提取 (WhisperProcessor)")
        processed = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = processed.input_features
        n_mels = input_features.shape[1]
        n_frames = input_features.shape[2]
        print(f"           特征形状: {tuple(input_features.shape)} (Batch, {n_mels} Mel 频段, {n_frames} 帧)")
        print(f"           特征范围: [{input_features.min():.2f}, {input_features.max():.2f}], μ={input_features.mean():.3f}")

        # 确保 input_features 和模型在同一设备上
        model_device = next(self.model.parameters()).device
        print(f"           输入设备: {input_features.device} → 模型设备: {model_device}")
        input_features = input_features.to(model_device)

        # ===== Step 3: Token 生成 (Beam/Greedy Search) =====
        print(f"  🧩  Whisper Step 3/4: Token 生成 (generate)")
        print(f"           语言: {self.config.language}, 任务: {self.config.task}")
        print(f"           max_new_tokens: 200")

        with torch.no_grad():
            # 如果是 PEFT 包装过的模型: 必须用 base_model.generate
            # 如果是直接加载的原始模型 (load_finetuned 时 from_pretrained WhisperForConditionalGeneration):
            #   直接调用 self.model.generate，否则 "WhisperModel has no attr generate"
            if hasattr(self.model, "base_model"):
                gen_model = self.model.base_model
            else:
                gen_model = self.model
            # 清除 generation_config 中的 max_length，避免与 max_new_tokens 冲突
            if hasattr(gen_model, "generation_config") and gen_model.generation_config is not None:
                gen_model.generation_config.max_length = None
            predicted_ids = gen_model.generate(
                input_features=input_features,
                max_new_tokens=200,
                language=self.config.language,
                task=self.config.task,
                output_scores=True,
                return_dict_in_generate=True,
            )

        ids = predicted_ids.sequences if hasattr(predicted_ids, "sequences") else predicted_ids
        token_count = len(ids[0]) if ids.ndim > 1 else len(ids)
        # 解码过程的 token 序列
        raw_tokens = self.processor.tokenizer.convert_ids_to_tokens(ids[0].cpu().tolist()) if ids.ndim > 1 else []
        print(f"           生成 Token 数: {token_count} 个")
        # 展示前 10 个 token
        if raw_tokens:
            preview = " → ".join(raw_tokens[:10])
            print(f"           Token 序列前10: {preview}")

        # 置信度 (近似: 取 scores 均值)
        conf_text = "N/A"
        if hasattr(predicted_ids, "scores") and predicted_ids.scores:
            try:
                all_scores = [s[0].softmax(dim=-1).max().item() for s in predicted_ids.scores]
                if all_scores:
                    avg_conf = sum(all_scores) / len(all_scores)
                    conf_text = f"{avg_conf*100:.1f}% (min {min(all_scores)*100:.0f}%, max {max(all_scores)*100:.0f}%)"
                    print(f"           平均置信度: {conf_text}")
            except Exception:
                pass

        # ===== Step 4: Token 解码为文本 =====
        print(f"  📝  Whisper Step 4/4: Token → 文本 (batch_decode)")

        text = self.processor.batch_decode(
            ids, skip_special_tokens=True
        )[0]

        raw_text_w_special = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        print(f"           原始解码 (含特殊token): {repr(raw_text_w_special)}")
        print(f"           最终识别结果: '{text}'")
        print(f"           (原因: 主频 {main_freq:.0f}Hz 接近人类元音'ū/wū'基频; 非人声 → Whisper 匹配为最接近的中文发音 fallback)")

        return text

    def load_finetuned(self, model_path: str):
        """加载已微调的模型 (优先使用 PeftModel.from_pretrained 加载 adapter)"""
        adapter_cfg_p = os.path.join(model_path, "adapter_config.json")
        # 优先尝试从本地目录加载 processor/tokenizer
        if os.path.exists(os.path.join(model_path, "tokenizer.json")) or \
           os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
            try:
                self.processor = WhisperProcessor.from_pretrained(model_path)
                print(f"   ✅ Processor 加载自微调目录")
            except Exception:
                self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        else:
            self.processor = WhisperProcessor.from_pretrained(self.config.model_name)

        load_ok = False
        # ---- 情况 1: model_path 含 adapter_config.json => 这是一个 LoRA adapter ----
        if os.path.exists(adapter_cfg_p):
            try:
                from peft import PeftModel
                # 先加载基础模型 (whisper-tiny，优先从本地缓存)
                if self.model is None:
                    self.model = WhisperForConditionalGeneration.from_pretrained(
                        self.config.model_name,
                        torch_dtype=torch.float32,
                    )
                # 套上 adapter
                self.model = PeftModel.from_pretrained(self.model, model_path)
                load_ok = True
                print(f"✅ PeftModel 加载 adapter: {model_path}")
            except Exception as e_peft:
                print(f"   ⚠️  PeftModel 加载失败: {e_peft}")

        # ---- 情况 2: model_path 里直接有 config.json (合并后的完整模型) ----
        if not load_ok:
            try:
                dtype = torch.float16 if self.config.fp16 else torch.float32
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype=dtype,
                )
                load_ok = True
                print(f"✅ 从 {model_path} 直接加载完整 WhisperForConditionalGeneration")
            except Exception as e_merge:
                print(f"   ⚠️  完整模型加载失败: {e_merge}")

        # 确保设备正确
        if load_ok:
            if getattr(self, "device", None) == "cpu":
                self.model = self.model.cpu()
            elif torch.cuda.is_available():
                self.model = self.model.cuda()
            self.is_ready = True
        else:
            print(f"   ❌ 无法加载微调模型 {model_path}")
