"""
LLM LoRA 微调模块: 微调大语言模型实现宠物声音理解 + 人宠翻译
任务:
  1. 宠物声音描述 → 情绪/意图解读
  2. 人类语言 → 宠物声音生成指令

支持自动检测硬件: Mac CPU / Mac MPS / GPU (V100 32GB / 4090 24GB)
支持网络检测: 网络不可用时自动降级
"""

import os
import glob
import signal
import time
import json

# 必须在导入 transformers 之前设置，覆盖 whisper_finetune.py 中的离线模式
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# 设置国内镜像源
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
import json
import socket
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# Fix: handle potential conflict between local datasets.py (e.g. MPViT-main/datasets.py)
# and the HuggingFace datasets package. Prefer the installed package.
import site as _site
import importlib.util as _ilu
import importlib.machinery as _imach

_datasets_spec = _ilu.find_spec("datasets")
if _datasets_spec and _datasets_spec.origin and "MPViT" in str(_datasets_spec.origin):
    # Found wrong datasets module (local file), force import from site-packages
    _sp_paths = []
    try:
        _sp_paths.extend(_site.getsitepackages())
    except Exception:
        pass
    try:
        _sp_paths.append(_site.getusersitepackages())
    except Exception:
        pass

    _hf_datasets_path = None
    for _sp in _sp_paths:
        _candidate = os.path.join(_sp, "datasets", "__init__.py")
        if os.path.isfile(_candidate):
            _hf_datasets_path = _candidate
            break

    if _hf_datasets_path:
        # Load the correct datasets package directly
        _loader = _imach.SourceFileLoader("datasets", _hf_datasets_path)
        _spec = _ilu.spec_from_loader("datasets", _loader)
        _datasets_module = _ilu.module_from_spec(_spec)
        sys.modules["datasets"] = _datasets_module
        _loader.exec_module(_datasets_module)

        # Now import Dataset from the correctly loaded module
        HFDataset = _datasets_module.Dataset
    else:
        from datasets import Dataset as HFDataset
else:
    from datasets import Dataset as HFDataset


# ============ 网络检测 ============

def check_network(timeout: int = 3) -> bool:
    """检测是否可以连接到 HuggingFace"""
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("huggingface.co", 443))
        s.close()
        return True
    except Exception:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("hf-mirror.com", 443))
            s.close()
            return True
        except Exception:
            return False


# ============ 硬件检测与自动配置 ============

def detect_device() -> Tuple[str, Dict]:
    """检测当前硬件设备信息"""
    info = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_memory_gb": 0,
        "device_name": "CPU",
        "platform": sys.platform,
        "is_linux": sys.platform.startswith("linux"),
        "is_mac": sys.platform == "darwin",
    }

    # 检测 CUDA
    if torch.cuda.is_available():
        info["gpu_available"] = True
        info["device"] = "cuda"
        info["gpu_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)

        # 获取显存大小
        for i in range(info["gpu_count"]):
            mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
            info["gpu_memory_gb"] = max(info["gpu_memory_gb"], mem)

    # 检测 MPS (Mac)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info["device"] = "mps"
        info["device_name"] = "Apple Silicon (MPS)"

    # CPU
    else:
        info["device"] = "cpu"
        info["device_name"] = "CPU"

    return info


def get_machine_config(machine_type: str = "auto") -> LLMFineTuneConfig:
    """
    根据硬件类型返回最优配置

    Args:
        machine_type: "auto" | "cpu" | "mps" | "gpu_8g" | "gpu_16g" | "gpu_24g" | "gpu_32g"

    Returns:
        LLMFineTuneConfig 实例
    """
    if machine_type == "auto":
        device_info = detect_device()
        print(f"🤖 自动检测硬件: {device_info['device_name']}")
        print(f"   设备类型: {device_info['device']}")
        print(f"   操作系统: {'Linux' if device_info['is_linux'] else 'macOS' if device_info['is_mac'] else '其他'}")
        if device_info["gpu_memory_gb"] > 0:
            print(f"   显存: {device_info['gpu_memory_gb']:.1f} GB")

        # 根据检测结果选择配置
        if device_info["device"] == "cuda":
            if device_info["gpu_memory_gb"] >= 24:
                machine_type = "gpu_24g"
            elif device_info["gpu_memory_gb"] >= 16:
                machine_type = "gpu_16g"
            elif device_info["gpu_memory_gb"] > 0:
                machine_type = "gpu_8g"
            else:
                machine_type = "cpu"
        elif device_info["device"] == "mps":
            # Apple Silicon MPS: 使用 M1/M2/M3 统一内存
            machine_type = "mps"
        else:
            machine_type = "cpu"

    # 预设配置
    configs = {
        # Mac CPU 快速验证 (135M 参数)
        "cpu": LLMFineTuneConfig(
            base_model="HuggingFaceTB/SmolLM2-135M-Instruct",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            max_seq_length=512,
            save_steps=50,
            eval_steps=50,
            bf16=False,  # CPU 用 float32
            output_dir="./checkpoints/llm_cpu",
        ),

        # Apple Silicon MPS (M1/M2/M3, 统一内存)
        "mps": LLMFineTuneConfig(
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            max_seq_length=1024,
            save_steps=50,
            eval_steps=50,
            bf16=False,  # MPS 用 float16 (在 _load_model 中处理)
            output_dir="./checkpoints/llm_mps",
        ),

        # GPU 8GB (T4/P100) - 最高精度可用模型
        "gpu_8g": LLMFineTuneConfig(
            base_model="Qwen/Qwen2.5-1.5B-Instruct",
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=10,
            max_seq_length=1024,
            save_steps=200,
            eval_steps=200,
            bf16=True,
            output_dir="./checkpoints/llm_gpu8g",
        ),

        # GPU 16GB (V100 16GB/A10) - 升级到 3B 模型
        "gpu_16g": LLMFineTuneConfig(
            base_model="Qwen/Qwen2.5-3B-Instruct",
            lora_r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=15,
            max_seq_length=2048,
            save_steps=200,
            eval_steps=200,
            bf16=True,
            output_dir="./checkpoints/llm_gpu16g",
        ),

        # GPU 24GB (RTX 4090) - 7B 模型, 更高配置
        "gpu_24g": LLMFineTuneConfig(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=15,
            max_seq_length=2048,
            save_steps=200,
            eval_steps=200,
            bf16=True,
            output_dir="./checkpoints/llm_gpu24g",
        ),

        # GPU 32GB (V100 32GB) - 7B 模型, 最大化利用显存
        "gpu_32g": LLMFineTuneConfig(
            base_model="Qwen/Qwen2.5-7B-Instruct",
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.05,
            learning_rate=2e-4,
            batch_size=16,
            gradient_accumulation_steps=1,
            num_train_epochs=20,
            max_seq_length=4096,
            save_steps=200,
            eval_steps=200,
            bf16=True,
            output_dir="./checkpoints/llm_gpu32g",
        ),
    }

    if machine_type not in configs:
        raise ValueError(
            f"未知的机器类型: {machine_type}\n"
            f"可选类型: {list(configs.keys())}"
        )

    config = configs[machine_type]
    print(f"\n📦 使用配置: {machine_type}")
    print(f"   模型: {config.base_model}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   训练轮数: {config.num_train_epochs}")
    print(f"   输出目录: {config.output_dir}")
    print()

    return config


# ============ 核心配置与模型 ============


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
        self.is_ready = False
        self._load_model()

    def _load_model(self):
        """加载基础模型 - 自动适配设备 + 网络检测"""
        device_info = detect_device()
        self.device = device_info["device"]

        print(f"加载基础模型: {self.config.base_model}")
        print(f"使用设备: {self.device} ({device_info['device_name']})")

        # 检测网络连接
        network_ok = check_network()
        print(f"网络状态: {'✅ 已连接' if network_ok else '❌ 未连接'}")

        # 选择合适的 torch dtype
        if self.device == "cuda" and self.config.bf16:
            torch_dtype = torch.bfloat16
        elif self.device == "mps":
            torch_dtype = torch.float16  # MPS 支持 float16
        else:
            torch_dtype = torch.float32  # CPU 默认 float32

        # 尝试加载 tokenizer 和 model
        try:
            if not network_ok:
                # 网络不可用: 尝试本地缓存，失败则报错
                print("⚠️  网络不可用，尝试从本地缓存加载...")
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._try_load_offline(torch_dtype)
                return

            # 网络可用: 正常加载
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            self._do_load(torch_dtype)

        except Exception as e:
            if network_ok:
                # 网络正常但加载失败，可能是模型不存在
                print(f"❌ 加载模型失败: {e}")
                print("💡 可能的原因:")
                print("   1. 模型名称错误")
                print("   2. 网络波动，请重试")
                print("   3. 模型需要认证 (Gated)")
                print(f"\n💡 解决方法:")
                print(f"   - 检查模型名称: {self.config.base_model}")
                print(f"   - 手动下载后使用本地路径: --model /path/to/local/model")
                print(f"   - 尝试其他模型: python scripts/train_llm.py --auto --model 'SmolLM2/SmolLM2-135M-Instruct'")
                raise
            else:
                # 网络不可用且本地也没有缓存
                self.is_ready = False
                print(f"\n❌ 网络不可用且本地无模型缓存！")
                print(f"\n💡 解决方案:")
                print(f"   方案1: 连接网络后重试 (推荐)")
                print(f"     python scripts/train_llm.py --auto")
                print(f"\n   方案2: 在另一台有网络的机器上下载模型，然后复制到本机")
                print(f"     # 下载缓存目录: ~/.cache/huggingface/hub/")
                print(f"\n   方案3: 使用已下载的 Whisper 模型做测试 (不支持 LLM 微调)")
                print(f"\n   方案4: 指定其他本地已缓存的模型路径")
                raise RuntimeError(
                    f"无法加载模型 {self.config.base_model}：网络不可用且本地无缓存。\n"
                    f"请连接网络后重试，或手动下载模型。"
                )

    def _try_load_offline(self, torch_dtype):
        """尝试离线加载模型"""
        try:
            self._do_load(torch_dtype)
        except Exception:
            self.is_ready = False
            raise RuntimeError(
                f"本地无 {self.config.base_model} 缓存，且网络不可用。\n"
                f"请连接网络后重试: python scripts/train_llm.py --auto"
            )

    def _do_load(self, torch_dtype):
        """执行实际的模型加载"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 根据设备选择加载方式
        if self.device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to("cpu")
        elif self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
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
        self.is_ready = True

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

    def train(
        self,
        train_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        resume_from: Optional[str] = None,
    ):
        """启动 LoRA 微调训练

        Args:
            train_data: 训练数据
            eval_data: 验证数据 (可选)
            resume_from: 指定从哪个 checkpoint 恢复训练 (可选)
                         - 传具体路径: 从指定 checkpoint 恢复
                         - 传 "auto": 自动检测最近的 checkpoint 恢复
                         - 不传: 从头开始训练
        """
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

        # ========== 优雅中断保存 ==========
        save_requested = [False]

        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n\n{'='*50}")
            print(f"⚠️  收到 {sig_name} 信号，正在保存 checkpoint...")
            print(f"   请耐心等待，保存完成后将自动退出")
            print(f"   再次按 Ctrl+C 将强制退出 (可能丢失进度)")
            print(f"{'='*50}")
            save_requested[0] = True
            # 恢复默认信号处理器，第二次 Ctrl+C 直接退出
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        # 注册信号处理器
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)

        # ========== 断点续训逻辑 ==========
        resume_checkpoint = self._resolve_resume_checkpoint(resume_from)

        if resume_checkpoint:
            print(f"\n{'='*50}")
            print(f"🔄 检测到 checkpoint，从断点恢复训练")
            print(f"   恢复路径: {resume_checkpoint}")
            print(f"{'='*50}\n")
        else:
            if self._find_latest_checkpoint():
                print(f"\n💡 提示: 检测到已有 checkpoint，可用 --resume auto 恢复训练")
                print(f"   最新 checkpoint: {self._find_latest_checkpoint()}")
                print(f"   或指定路径: --resume {self._find_latest_checkpoint()}\n")

        print(f"💡 提示: 训练中按 Ctrl+C 可优雅中断并自动保存")
        print(f"   checkpoint 间隔: 每 {self.config.save_steps} 步")
        print(f"   最多保留: {training_args.save_total_limit} 个 checkpoint")

        try:
            print("开始 LLM LoRA 微调训练...")
            trainer.train(resume_from_checkpoint=resume_checkpoint)

            # 训练正常完成
            save_path = os.path.join(self.config.output_dir, "final")
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f"✅ 训练完成! LoRA adapter 已保存到: {save_path}")

        except (KeyboardInterrupt, SystemExit):
            if save_requested[0]:
                print(f"\n{'='*50}")
                print(f"💾 正在保存当前状态...")
                save_path = os.path.join(self.config.output_dir, "checkpoint-interrupted")
                trainer.save_checkpoint(save_path)
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # 写入中断信息
                status_file = os.path.join(self.config.output_dir, "interrupted_info.json")
                status = {
                    "interrupted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "checkpoint_path": save_path,
                    "message": "训练已中断，可使用此 checkpoint 恢复",
                    "resume_command": f"python scripts/train_llm.py --auto --resume {save_path}",
                }
                with open(status_file, "w") as f:
                    json.dump(status, f, ensure_ascii=False, indent=2)

                print(f"✅ checkpoint 已保存到: {save_path}")
                print(f"📝 恢复命令: python scripts/train_llm.py --auto --resume {save_path}")
                print(f"📝 或自动恢复: python scripts/train_llm.py --auto --resume auto")
                print(f"{'='*50}\n")
                raise SystemExit(0)
            else:
                # 非预期的键盘中断（第二次 Ctrl+C）
                print(f"\n\n⚠️  强制退出，当前进度可能已丢失！")
                raise

        finally:
            # 恢复原始信号处理器
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

        return save_path

    @staticmethod
    def check_training_status(output_dir: str) -> Dict:
        """检查训练状态和已有 checkpoint

        Args:
            output_dir: 训练输出目录 (如 ./checkpoints/llm_mps)

        Returns:
            状态信息字典
        """
        status = {
            "output_dir": output_dir,
            "exists": os.path.exists(output_dir),
            "checkpoints": [],
            "final_exists": False,
            "latest_checkpoint": None,
            "interrupted_info": None,
        }

        if not status["exists"]:
            return status

        # 查找所有 checkpoint
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=os.path.getmtime,
        )

        for ckpt_dir in checkpoint_dirs:
            ckpt_info = {
                "path": ckpt_dir,
                "name": os.path.basename(ckpt_dir),
                "modified": time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(os.path.getmtime(ckpt_dir)),
                ),
            }
            # 尝试读取训练状态
            state_file = os.path.join(ckpt_dir, "trainer_state.json")
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        state = json.load(f)
                    ckpt_info["global_step"] = state.get("global_step", 0)
                    ckpt_info["epoch"] = state.get("epoch", 0)
                except Exception:
                    pass
            status["checkpoints"].append(ckpt_info)

        if status["checkpoints"]:
            status["latest_checkpoint"] = status["checkpoints"][-1]

        # 检查 final 是否存在
        final_dir = os.path.join(output_dir, "final")
        status["final_exists"] = os.path.exists(final_dir)

        # 检查中断信息
        interrupted_file = os.path.join(output_dir, "interrupted_info.json")
        if os.path.exists(interrupted_file):
            try:
                with open(interrupted_file, "r") as f:
                    status["interrupted_info"] = json.load(f)
            except Exception:
                pass

        return status

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找 output_dir 中最新的 checkpoint 路径"""
        checkpoint_dirs = sorted(
            glob.glob(os.path.join(self.config.output_dir, "checkpoint-*")),
            key=os.path.getmtime,
        )
        return checkpoint_dirs[-1] if checkpoint_dirs else None

    def _resolve_resume_checkpoint(self, resume_from: Optional[str]) -> Optional[str]:
        """解析 resume_from 参数，返回有效的 checkpoint 路径"""
        if resume_from is None:
            return None

        if resume_from == "auto":
            latest = self._find_latest_checkpoint()
            if latest:
                return latest
            print("⚠️ 未找到已有 checkpoint，将从头开始训练")
            return None

        # 指定了具体路径
        if os.path.isdir(resume_from):
            # 检查是否是有效的 checkpoint
            if os.path.exists(os.path.join(resume_from, "trainer_state.json")):
                return resume_from
            else:
                print(f"⚠️ 指定路径 {resume_from} 不是有效的 checkpoint (缺少 trainer_state.json)")
                return None
        else:
            print(f"⚠️ 指定路径 {resume_from} 不存在，将从头开始训练")
            return None

    def inference(
        self,
        user_input: str,
        system_prompt: str = "你是一个宠物语言翻译专家。",
        max_new_tokens: int = 512,
    ) -> str:
        """推理 - 自动适配设备"""
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

        # 根据设备移动输入
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        elif self.device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}

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
        """加载已微调的 LoRA adapter - 自动适配设备"""
        from peft import PeftModel

        # 选择合适的 torch dtype
        if self.device == "cuda" and self.config.bf16:
            torch_dtype = torch.bfloat16
        elif self.device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # 先加载基础模型
        if self.device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        elif self.device == "mps":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            self.model = self.model.to("mps")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        # 加载 LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.model = self.model.merge_and_unload()

        # 确保模型在正确设备上
        if self.device == "cuda":
            self.model = self.model.cuda()
        elif self.device == "mps":
            self.model = self.model.to("mps")

        print(f"✅ 已加载微调模型: {adapter_path}")
        self.is_ready = True
