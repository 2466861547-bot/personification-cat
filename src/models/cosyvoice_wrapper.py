"""CosyVoice2-0.5B 自包含推理模块

完整实现 CosyVoice2 的推理管线:
  文本 → LLM(Qwen2) → 语音token → Flow Matching → Mel频谱 → HiFT → 音频波形

模型文件结构 (iic/CosyVoice2-0.5B):
  - llm.pt            : Qwen2 LLM 权重 (文本→语音token)
  - flow.pt           : Flow Matching 模型权重 (语音token→Mel)
  - hift.pt           : HiFT HiFi-GAN 权重 (Mel→音频)
  - campplus.onnx     : Speaker Embedding 模型 (参考音频→声纹)
  - speech_tokenizer_v2.onnx : 语音tokenizer (Mel→离散token)
  - flow.decoder.estimator.fp32.onnx : Flow Decoder 估算器

用法:
    model = CosyVoice2(model_dir)
    audio = model.inference_zero_shot("你好世界", "参考文本", "ref.wav")
    audio = model.inference_cross_lingual("Hello world", "ref.wav")
"""

import os
import re
import struct
import logging
from typing import Optional, List, Generator, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("cosyvoice")

SAMPLE_RATE = 24000
HOP_SIZE = 480
MEL_DIM = 80
MAX_TEXT_LEN = 50


def _strip_hydra_tags(yaml_text: str) -> dict:
    """去除 Hydra/OmegaConf 自定义标签，提取纯配置值"""
    result = {}
    for line in yaml_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Skip lines with Hydra tags
        if any(tag in stripped for tag in ["!apply:", "!new:", "!ref ", "!name:"]):
            continue
        # Extract key: value pairs for simple values
        if ":" in stripped and not stripped.endswith(":"):
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val and key and not val.startswith("["):
                try:
                    if "." in val:
                        result[key] = float(val)
                    elif val == "True":
                        result[key] = True
                    elif val == "False":
                        result[key] = False
                    elif val == "null":
                        result[key] = None
                    else:
                        result[key] = int(val)
                except ValueError:
                    result[key] = val
    return result


class CosyVoice2:
    """CosyVoice 2.0.5B 完整推理实现"""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.config = self._load_config()
        self.sample_rate = int(self.config.get("sample_rate", SAMPLE_RATE))

        self.llm = None
        self.flow_encoder = None
        self.flow_decoder_session = None
        self.spk_embed_affine = None
        self.hift = None
        self.campplus_session = None
        self.speech_tokenizer_session = None
        self.tokenizer = None

        self._load_all()

    def _load_config(self) -> dict:
        config_path = os.path.join(self.model_dir, "cosyvoice2.yaml")
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                raw = f.read()
            config = _strip_hydra_tags(raw)
        return config

    def _load_all(self):
        """加载所有模型组件"""
        logger.info("加载 LLM (Qwen2) ...")
        self._load_llm()

        logger.info("加载 Flow 模型 ...")
        self._load_flow()

        logger.info("加载 HiFT (HiFi-GAN) ...")
        self._load_hift()

        logger.info("加载 Campplus (Speaker Embedding) ...")
        self._load_campplus()

        logger.info("加载 Speech Tokenizer ...")
        self._load_speech_tokenizer()

        logger.info("加载文本 Tokenizer ...")
        self._load_text_tokenizer()

        logger.info("✅ CosyVoice2 全部组件加载完成")

    def _load_llm(self):
        """加载 Qwen2 LLM 用于文本→语音token"""
        llm_path = os.path.join(self.model_dir, "llm.pt")
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"LLM 权重不存在: {llm_path}")

        state_dict = torch.load(llm_path, map_location=self.device, weights_only=True)

        # 推断模型结构
        num_layers = 0
        for key in state_dict:
            m = re.match(r'llm\.model\.model\.layers\.(\d+)\.', key)
            if m:
                num_layers = max(num_layers, int(m.group(1)) + 1)

        # 从权重形状推断参数
        embed_weight = state_dict.get("llm.model.model.embed_tokens.weight")
        vocab_size, hidden_size = embed_weight.shape if embed_weight is not None else (151665, 896)

        # Attention heads from q_proj
        q_proj = state_dict.get("llm.model.model.layers.0.self_attn.q_proj.weight")
        num_attention_heads = q_proj.shape[0] // (q_proj.shape[1] // 8) if q_proj is not None else 8

        # FFN intermediate size
        gate_proj = state_dict.get("llm.model.model.layers.0.mlp.gate_proj.weight")
        intermediate_size = gate_proj.shape[0] if gate_proj is not None else hidden_size * 3

        # Speech vocab size
        speech_embed = state_dict.get("speech_embedding.weight")
        speech_vocab_size = speech_embed.shape[0] if speech_embed is not None else 6564

        # Text vocab size (lm_head)
        lm_head_weight = state_dict.get("llm.model.lm_head.weight")
        text_vocab_size = lm_head_weight.shape[0] if lm_head_weight is not None else vocab_size

        self.llm = _Qwen2LLM(
            text_vocab_size=text_vocab_size,
            speech_vocab_size=speech_vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
        )

        # 加载权重到模型
        model_sd = {}
        for key, val in state_dict.items():
            # Map keys from checkpoint format to model format
            if key == "llm.model.model.embed_tokens.weight":
                model_sd["embedding.weight"] = val
            elif key == "llm.model.lm_head.weight":
                model_sd["lm_head.weight"] = val
            elif key == "speech_embedding.weight":
                model_sd["speech_embedding.weight"] = val
            elif key == "llm_decoder.weight":
                model_sd["speech_head.weight"] = val
            elif key == "llm_decoder.bias":
                model_sd["speech_head.bias"] = val
            elif key == "llm.model.model.norm.weight":
                model_sd["transformer.norm.weight"] = val
            elif key.startswith("llm.model.model.layers."):
                # Map: llm.model.model.layers.N.self_attn.q_proj.weight -> transformer.layers.N.q_proj.weight
                # Map: llm.model.model.layers.N.self_attn.k_proj.weight -> transformer.layers.N.k_proj.weight
                # etc.
                layer_idx = re.search(r'layers\.(\d+)\.', key)
                if layer_idx:
                    idx = layer_idx.group(1)
                    if 'self_attn.q_proj' in key:
                        new_key = f'transformer.layers.{idx}.q_proj.weight' if 'weight' in key else f'transformer.layers.{idx}.q_proj.bias'
                    elif 'self_attn.k_proj' in key:
                        new_key = f'transformer.layers.{idx}.k_proj.weight' if 'weight' in key else f'transformer.layers.{idx}.k_proj.bias'
                    elif 'self_attn.v_proj' in key:
                        new_key = f'transformer.layers.{idx}.v_proj.weight' if 'weight' in key else f'transformer.layers.{idx}.v_proj.bias'
                    elif 'self_attn.o_proj' in key:
                        new_key = f'transformer.layers.{idx}.o_proj.weight' if 'weight' in key else f'transformer.layers.{idx}.o_proj.bias'
                    elif 'mlp.gate_proj' in key:
                        new_key = f'transformer.layers.{idx}.gate_proj.weight'
                    elif 'mlp.up_proj' in key:
                        new_key = f'transformer.layers.{idx}.up_proj.weight'
                    elif 'mlp.down_proj' in key:
                        new_key = f'transformer.layers.{idx}.down_proj.weight'
                    elif 'input_layernorm' in key:
                        new_key = f'transformer.layers.{idx}.norm1.weight'
                    elif 'post_attention_layernorm' in key:
                        new_key = f'transformer.layers.{idx}.norm2.weight'
                    else:
                        continue
                    model_sd[new_key] = val
            # Skip llm_embedding.weight (special tokens, shape [2, 896])

        self.llm.load_state_dict(model_sd, strict=False)
        self.llm = self.llm.to(self.device)
        self.llm.eval()
        logger.info(f"  LLM 加载成功: {num_layers} layers, hidden={hidden_size}, heads={num_attention_heads}")

    def _load_flow(self):
        """加载 Flow Matching 模型"""
        flow_path = os.path.join(self.model_dir, "flow.pt")
        if not os.path.exists(flow_path):
            flow_path = os.path.join(self.model_dir, "flow.cache.pt")
        if not os.path.exists(flow_path):
            raise FileNotFoundError(f"Flow 权重不存在: {flow_path}")

        state_dict = torch.load(flow_path, map_location=self.device, weights_only=True)

        # Build encoder state dict with key mapping
        enc_state = {}
        for key, val in state_dict.items():
            if key.startswith("decoder."):
                continue
            if key.startswith("spk_embed"):
                continue

            new_key = key

            if key == "input_embedding.weight":
                new_key = "input_embedding.weight"
            elif key.startswith("encoder.embed.out."):
                idx = key.split(".")[-2]
                suffix = key.split(".")[-1]
                new_key = f"embed.{idx}.{suffix}"
            elif key.startswith("encoder.pre_lookahead_layer."):
                idx = key.split(".")[-2]
                suffix = key.split(".")[-1]
                idx_num = 0 if idx == "conv1" else 1
                new_key = f"pre_lookahead.{idx_num}.{suffix}"
            elif key.startswith("encoder.encoders."):
                parts = key.split(".")
                layer_idx = parts[2]
                sub_type = parts[3]
                suffix = parts[-1]

                if sub_type == "feed_forward":
                    w_idx = int(parts[4].replace("w_", ""))
                    new_idx = w_idx * 2 - 2  # w_1→0, w_2→2
                    new_key = f"encoders.{layer_idx}.feed_forward.{new_idx}.{suffix}"
                elif sub_type == "norm_ff":
                    new_key = f"encoders.{layer_idx}.norm_ff.{suffix}"
                elif sub_type == "norm_mha":
                    new_key = f"encoders.{layer_idx}.norm_mha.{suffix}"
                elif sub_type == "self_attn":
                    attn_part = parts[4]
                    if attn_part == "linear_q":
                        new_key = f"encoders.{layer_idx}.self_attn.linear_q.{suffix}"
                    elif attn_part == "linear_k":
                        new_key = f"encoders.{layer_idx}.self_attn.linear_k.{suffix}"
                    elif attn_part == "linear_v":
                        new_key = f"encoders.{layer_idx}.self_attn.linear_v.{suffix}"
                    elif attn_part == "linear_out":
                        new_key = f"encoders.{layer_idx}.self_attn.linear_out.{suffix}"
                    elif attn_part == "linear_pos":
                        new_key = f"encoders.{layer_idx}.self_attn.linear_pos.{suffix}"
                    elif attn_part == "pos_bias_u":
                        new_key = f"encoders.{layer_idx}.self_attn.pos_bias_u"
                    elif attn_part == "pos_bias_v":
                        new_key = f"encoders.{layer_idx}.self_attn.pos_bias_v"
            elif key.startswith("encoder.up_embed."):
                idx = key.split(".")[-2]
                suffix = key.split(".")[-1]
                new_key = f"up_embed.{idx}.{suffix}"
            elif key.startswith("encoder.up_encoders."):
                parts = key.split(".")
                layer_idx = parts[2]
                sub_type = parts[3]
                suffix = parts[-1]

                if sub_type == "feed_forward":
                    w_idx = int(parts[4].replace("w_", ""))
                    new_idx = w_idx * 2 - 2  # w_1→0, w_2→2
                    new_key = f"up_encoders.{layer_idx}.feed_forward.{new_idx}.{suffix}"
                elif sub_type == "norm_ff":
                    new_key = f"up_encoders.{layer_idx}.norm_ff.{suffix}"
                elif sub_type == "norm_mha":
                    new_key = f"up_encoders.{layer_idx}.norm_mha.{suffix}"
                elif sub_type == "self_attn":
                    attn_part = parts[4]
                    if attn_part == "linear_q":
                        new_key = f"up_encoders.{layer_idx}.self_attn.linear_q.{suffix}"
                    elif attn_part == "linear_k":
                        new_key = f"up_encoders.{layer_idx}.self_attn.linear_k.{suffix}"
                    elif attn_part == "linear_v":
                        new_key = f"up_encoders.{layer_idx}.self_attn.linear_v.{suffix}"
                    elif attn_part == "linear_out":
                        new_key = f"up_encoders.{layer_idx}.self_attn.linear_out.{suffix}"
                    elif attn_part == "linear_pos":
                        new_key = f"up_encoders.{layer_idx}.self_attn.linear_pos.{suffix}"
                    elif attn_part == "pos_bias_u":
                        new_key = f"up_encoders.{layer_idx}.self_attn.pos_bias_u"
                    elif attn_part == "pos_bias_v":
                        new_key = f"up_encoders.{layer_idx}.self_attn.pos_bias_v"
            elif key.startswith("encoder.up_layer."):
                suffix = key.split(".")[-1]
                new_key = f"up_layer_conv.{suffix}"
            elif key.startswith("encoder_proj."):
                suffix = key.split(".")[-1]
                new_key = f"encoder_proj.{suffix}"
            elif key.startswith("encoder.after_norm."):
                suffix = key.split(".")[-1]
                new_key = f"after_norm.{suffix}"

            enc_state[new_key] = val

        # Create Flow Encoder
        self.flow_encoder = _FlowEncoder()
        self.flow_encoder.load_state_dict(enc_state, strict=False)
        self.flow_encoder = self.flow_encoder.to(self.device)
        self.flow_encoder.eval()

        # Load speaker embedding affine layer (192→80)
        self.spk_embed_affine = nn.Linear(192, MEL_DIM)
        spk_state = {}
        for key, val in state_dict.items():
            if key.startswith("spk_embed_affine"):
                suffix = key.split(".")[-1]
                spk_state[suffix] = val
        if spk_state:
            self.spk_embed_affine.load_state_dict(spk_state, strict=False)
        self.spk_embed_affine = self.spk_embed_affine.to(self.device)
        self.spk_embed_affine.eval()

        # Load ONNX Flow Decoder
        flow_dec_onnx = os.path.join(self.model_dir, "flow.decoder.estimator.fp32.onnx")
        if os.path.exists(flow_dec_onnx):
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if self.device == "cuda":
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.flow_decoder_session = ort.InferenceSession(flow_dec_onnx, providers=providers)
                logger.info(f"  Flow Decoder (ONNX) 加载成功")
            except Exception as e:
                logger.warning(f"  Flow Decoder ONNX 加载失败: {e}")
                self.flow_decoder_session = None
        else:
            self.flow_decoder_session = None
            logger.warning("  Flow Decoder ONNX 不存在")

        logger.info(f"  Flow 加载成功: encoder + spk_embed + decoder(ONNX)")

    def _load_hift(self):
        """加载 HiFT HiFi-GAN"""
        hift_path = os.path.join(self.model_dir, "hift.pt")
        if not os.path.exists(hift_path):
            raise FileNotFoundError(f"HiFT 权重不存在: {hift_path}")

        state_dict = torch.load(hift_path, map_location=self.device, weights_only=True)

        self.hift = _HiFTGenerator()

        # Map parametrized weights: parametrizations.weight.original0/1 → param0/param1
        # Map activation alphas: activations1.{i}.alpha → activations1.{i}.alpha
        mapped_state = {}
        for key, val in state_dict.items():
            new_key = key

            # Handle weight parametrizations
            if 'parametrizations.weight.original0' in key:
                new_key = key.replace('parametrizations.weight.original0', 'param0')
            elif 'parametrizations.weight.original1' in key:
                new_key = key.replace('parametrizations.weight.original1', 'param1')

            mapped_state[new_key] = val

        self.hift.load_state_dict(mapped_state, strict=False)
        self.hift = self.hift.to(self.device)
        self.hift.eval()
        logger.info("  HiFT 加载成功")

    def _load_campplus(self):
        """加载 Campplus Speaker Embedding ONNX 模型"""
        onnx_path = os.path.join(self.model_dir, "campplus.onnx")
        if not os.path.exists(onnx_path):
            logger.warning(f"  Campplus ONNX 不存在: {onnx_path}")
            return

        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.campplus_session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"  Campplus ONNX 加载成功")
        except Exception as e:
            logger.warning(f"  Campplus 加载失败: {e}")
            self.campplus_session = None

    def _load_speech_tokenizer(self):
        """加载 Speech Tokenizer ONNX 模型"""
        onnx_path = os.path.join(self.model_dir, "speech_tokenizer_v2.onnx")
        if not os.path.exists(onnx_path):
            logger.warning(f"  Speech Tokenizer ONNX 不存在: {onnx_path}")
            return

        try:
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.speech_tokenizer_session = ort.InferenceSession(onnx_path, providers=providers)
            logger.info(f"  Speech Tokenizer ONNX 加载成功")
        except Exception as e:
            logger.warning(f"  Speech Tokenizer 加载失败: {e}")
            self.speech_tokenizer_session = None

    def _load_text_tokenizer(self):
        """加载文本 tokenizer (BPE/SentencePiece)"""
        try:
            import sentencepiece as spm
            for name in ["bpe.model", "tokenizer.model"]:
                path = os.path.join(self.model_dir, name)
                if os.path.exists(path):
                    self.tokenizer = spm.SentencePieceProcessor()
                    self.tokenizer.Load(path)
                    logger.info(f"  文本 Tokenizer 加载成功: {path}")
                    return

            # Check CosyVoice-BlankEN subdirectory
            blank_dir = os.path.join(self.model_dir, "CosyVoice-BlankEN")
            if os.path.isdir(blank_dir):
                tokenizer_path = os.path.join(blank_dir, "merges.txt")
                if os.path.exists(tokenizer_path):
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(blank_dir)
                    logger.info(f"  文本 Tokenizer (BPE) 加载成功")
                    return

            logger.warning("  文本 Tokenizer 未找到，使用字符级回退")
            self.tokenizer = None
        except Exception as e:
            logger.warning(f"  文本 Tokenizer 加载失败: {e}")
            self.tokenizer = None

    def _tokenize_text(self, text: str) -> List[int]:
        if self.tokenizer is not None:
            from transformers import PreTrainedTokenizerFast
            if isinstance(self.tokenizer, PreTrainedTokenizerFast):
                return self.tokenizer.encode(text, add_special_tokens=False)
            elif hasattr(self.tokenizer, "encode"):
                return self.tokenizer.encode(text, out_type=int)
            else:
                return self.tokenizer.encode(text)
        return [ord(c) for c in text]

    def _extract_speaker_embedding(self, audio_path: str) -> torch.Tensor:
        """从参考音频提取 speaker embedding"""
        import librosa
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)

        if self.campplus_session is not None:
            # 计算 mel-spectrogram
            mel = self._mel_spectrogram(audio)  # [80, T]

            # Campplus 输入: [batch, seq_len, 80]
            mel_input = mel.transpose(1, 0)[np.newaxis, :, :].astype(np.float32)  # [1, T, 80]
            outputs = self.campplus_session.run(None, {"input": mel_input})
            emb = outputs[0]  # [1, 192]
            spk_emb = torch.from_numpy(emb).float().to(self.device)
            spk_emb = F.normalize(spk_emb, dim=-1)
            return spk_emb
        else:
            # 降级：使用频谱统计
            mel = self._mel_spectrogram(audio)
            emb = torch.from_numpy(mel.mean(axis=1)).float().to(self.device)
            emb = F.normalize(emb.unsqueeze(0), dim=-1)
            return emb

    def _mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """计算 mel 频谱 (匹配 CosyVoice 参数)"""
        import librosa
        n_fft = 1920
        hop_size = HOP_SIZE
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_size,
            n_mels=MEL_DIM, fmin=0, fmax=8000, center=False,
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        return mel_db.astype(np.float32)

    def _text_to_speech_tokens(self, text: str) -> torch.Tensor:
        """使用 LLM 将文本转为语音 token 序列"""
        text_tokens = self._tokenize_text(text)
        text_tensor = torch.tensor([text_tokens], dtype=torch.long, device=self.device)

        with torch.no_grad():
            speech_tokens = self.llm.generate_speech(text_tensor, max_new_tokens=300)
        return speech_tokens

    def _flow_generate_mel(
        self,
        speech_tokens: torch.Tensor,
        spk_emb: torch.Tensor,
        inference_steps: int = 10,
    ) -> torch.Tensor:
        """使用 Flow Matching 生成 mel-spectrogram (ONNX Decoder)"""
        with torch.no_grad():
            B = 1
            T = speech_tokens.shape[1]

            # Flow Encoder: speech tokens → cond [B, 80, T]
            cond = self.flow_encoder(speech_tokens)  # [B, 80, T]

            # Speaker embedding affine: [B, 192] → [B, 80]
            spk_emb_80 = self.spk_embed_affine(spk_emb)  # [B, 80]

            # Flow matching: iteratively refine mel from noise
            x = torch.randn(B, MEL_DIM, T, device=self.device, dtype=self.dtype)
            mask = torch.ones(B, 1, T, device=self.device, dtype=self.dtype)

            for i in range(inference_steps):
                t_val = i / inference_steps
                t = torch.tensor([t_val] * B, device=self.device, dtype=self.dtype)

                # Run ONNX decoder
                x_np = x.cpu().numpy().astype(np.float32)
                mask_np = mask.cpu().numpy().astype(np.float32)
                cond_np = cond.cpu().numpy().astype(np.float32)
                spks_np = spk_emb_80.cpu().numpy().astype(np.float32)
                t_np = t.cpu().numpy().astype(np.float32)
                mu_np = cond_np  # mean = condition in CosyVoice2

                outputs = self.flow_decoder_session.run(
                    None,
                    {
                        "x": x_np,
                        "mask": mask_np,
                        "mu": mu_np,
                        "t": t_np,
                        "spks": spks_np,
                        "cond": cond_np,
                    }
                )

                pred = torch.from_numpy(outputs[0]).to(self.device).to(self.dtype)
                x = x + pred / inference_steps

        return x

    def _hift_decode(self, mel: torch.Tensor) -> np.ndarray:
        """使用 HiFT 将 mel 转为音频波形"""
        with torch.no_grad():
            # mel: [1, 80, T]
            audio = self.hift(mel)
            audio = audio.squeeze().cpu().numpy()
        return audio

    def inference_zero_shot(
        self,
        text: str,
        reference_text: str,
        reference_audio: str,
        inference_timesteps: int = 10,
    ) -> Generator[np.ndarray, None, None]:
        """Zero-shot 声纹克隆"""
        text = self._normalize_text(text)
        reference_text = self._normalize_text(reference_text)

        spk_emb = self._extract_speaker_embedding(reference_audio)
        speech_tokens = self._text_to_speech_tokens(text)
        mel = self._flow_generate_mel(speech_tokens, spk_emb, inference_timesteps)
        audio = self._hift_decode(mel)
        yield audio

    def inference_cross_lingual(
        self,
        text: str,
        reference_audio: str,
        inference_timesteps: int = 10,
    ) -> Generator[np.ndarray, None, None]:
        """跨语言声纹克隆"""
        text = self._normalize_text(text)
        spk_emb = self._extract_speaker_embedding(reference_audio)
        speech_tokens = self._text_to_speech_tokens(text)
        mel = self._flow_generate_mel(speech_tokens, spk_emb, inference_timesteps)
        audio = self._hift_decode(mel)
        yield audio

    def _normalize_text(self, text: str) -> str:
        import re
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text


# ============================================================
# Qwen2 LLM Implementation
# ============================================================

class _Qwen2LLM(nn.Module):
    def __init__(self, text_vocab_size, speech_vocab_size, hidden_size,
                 num_layers, num_attention_heads, intermediate_size):
        super().__init__()
        self.embedding = nn.Embedding(text_vocab_size, hidden_size)
        self.speech_embedding = nn.Embedding(speech_vocab_size, hidden_size)
        self.transformer = _Qwen2Transformer(
            hidden_size, num_layers, num_attention_heads, intermediate_size
        )
        self.lm_head = nn.Linear(hidden_size, text_vocab_size, bias=False)
        self.speech_head = nn.Linear(hidden_size, speech_vocab_size)

    def forward(self, input_ids, attention_mask=None, use_speech_embedding=False):
        if use_speech_embedding:
            x = self.speech_embedding(input_ids)
        else:
            x = self.embedding(input_ids)
        x = self.transformer(x, attention_mask)
        text_logits = self.lm_head(x)
        speech_logits = self.speech_head(x)
        return text_logits, speech_logits

    def generate_speech(self, input_ids, max_new_tokens=300):
        """自回归生成语音 token"""
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            text_logits, speech_logits = self.forward(generated)
            next_token = speech_logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == 0).all():
                break

        return generated[:, input_ids.shape[1]:]


class _Qwen2Transformer(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attention_heads, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([
            _Qwen2DecoderLayer(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.norm(x)


class _Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        # MQA: Q has multiple heads, K/V are shared (1 head)
        # q_proj: [896, 896] -> 7 heads × 128 dim
        # k_proj: [128, 896] -> 1 head × 128 dim (shared across Q heads)
        # v_proj: [128, 896] -> 1 head × 128 dim (shared)
        self.head_dim = 128
        self.num_heads = hidden_size // self.head_dim  # 896 // 128 = 7
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, 128, bias=True)
        self.v_proj = nn.Linear(hidden_size, 128, bias=True)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape
        residual = x
        x = self.norm1(x)

        # Q projection: [B, T, hidden] -> [B, num_heads, T, head_dim]
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # K,V projection: [B, T, hidden] -> [B, 1, T, 128] (multi-query)
        k = self.k_proj(x).unsqueeze(1)  # [B, 1, T, 128]
        v = self.v_proj(x).unsqueeze(1)  # [B, 1, T, 128]

        # Repeat K,V for each head
        k = k.expand(-1, self.num_heads, -1, -1)  # [B, num_heads, T, 128]
        v = v.expand(-1, self.num_heads, -1, -1)  # [B, num_heads, T, 128]

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, num_heads, T, T]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, num_heads, T, 128]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)

        x = residual + out

        residual = x
        x = self.norm2(x)
        # SwiGLU: gate_proj(x) * silu(up_proj(x)), then down_proj
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        x = residual + x
        return x


# ============================================================
# Flow Matching Implementation
# ============================================================

class _FlowEncoder(nn.Module):
    """Flow Encoder: speech token IDs → mel-space condition [B, 80, T]"""
    def __init__(self):
        super().__init__()
        self.input_embedding = nn.Embedding(6561, 512)
        self.embed = nn.ModuleList([
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        ])
        self.pre_lookahead = nn.ModuleList([
            nn.Conv1d(512, 512, 4, padding=0),
            nn.Conv1d(512, 512, 3, padding=0),
        ])
        self.encoders = nn.ModuleList([
            _ConformerBlock(512, 8, 2048)
            for _ in range(6)
        ])
        self.after_norm = nn.LayerNorm(512)
        self.up_embed = nn.ModuleList([
            nn.Linear(512, 512),
            nn.LayerNorm(512),
        ])
        self.up_encoders = nn.ModuleList([
            _ConformerBlock(512, 8, 2048)
            for _ in range(4)
        ])
        self.up_layer_conv = nn.Conv1d(512, 512, 5, padding=2)
        self.encoder_proj = nn.Linear(512, 80)

    def forward(self, x):
        """x: [B, T] speech token IDs → [B, 80, T] mel condition"""
        # Embed speech tokens
        x = self.input_embedding(x)  # [B, T, 512]

        # Embedding projection
        x = self.embed[0](x)
        x = self.embed[1](x)

        # Pre-lookahead (causal conv)
        x = x.transpose(1, 2)  # [B, 512, T]
        x = self.pre_lookahead[0](x)  # kernel=4, shrinks by 3
        x = self.pre_lookahead[1](x)  # kernel=3, shrinks by 2
        x = x.transpose(1, 2)  # [B, T', 512]

        # Conformer encoder
        for layer in self.encoders:
            x = layer(x)
        x = self.after_norm(x)

        # Up-embed projection
        x = self.up_embed[0](x)
        x = self.up_embed[1](x)

        # Up-encoder Conformer blocks
        for layer in self.up_encoders:
            x = layer(x)

        # Up-layer conv
        x = x.transpose(1, 2)  # [B, 512, T]
        x = self.up_layer_conv(x)
        x = x.transpose(1, 2)  # [B, T, 512]

        # Project to mel space
        x = self.encoder_proj(x)  # [B, T, 80]
        x = x.transpose(1, 2)  # [B, 80, T]
        return x


class _ConformerBlock(nn.Module):
    """Conformer block with relative positional bias"""
    def __init__(self, d_model, num_heads, linear_units):
        super().__init__()
        self.norm_ff = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, linear_units),
            nn.SiLU(),
            nn.Linear(linear_units, d_model),
        )
        self.norm_mha = nn.LayerNorm(d_model)
        self.self_attn = _RelPosMultiHeadAttention(d_model, num_heads)
        self.norm_ff2 = nn.LayerNorm(d_model)
        self.feed_forward2 = nn.Sequential(
            nn.Linear(d_model, linear_units),
            nn.SiLU(),
            nn.Linear(linear_units, d_model),
        )
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        # FFN
        x = x + 0.5 * self.feed_forward(self.norm_ff(x))
        # Self-attention
        x = x + self.self_attn(self.norm_mha(x))
        # FFN
        x = x + 0.5 * self.feed_forward2(self.norm_ff2(x))
        return self.norm_final(x)


class _RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional bias"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 64
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model)
        self.pos_bias_u = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.randn(num_heads, self.head_dim))

    def forward(self, x):
        B, T, C = x.shape
        q = self.linear_q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.linear_k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.linear_v(x).view(B, T, self.num_heads, self.head_dim)

        # Relative positional bias
        pos_ids = torch.arange(T, device=x.device)
        dist = pos_ids.unsqueeze(0) - pos_ids.unsqueeze(1)  # [T, T]
        # Clip to [-4, 4]
        dist = dist.clamp(-4, 4)
        # Compute positional encoding
        pos_enc = self.linear_pos(torch.eye(T, device=x.device).unsqueeze(0))  # Simplified

        # Simple relative position bias
        # [B, num_heads, T, T]
        attn = torch.matmul(q.transpose(1, 2), k.transpose(1, 2).transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.transpose(1, 2))  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.linear_out(out)


# ============================================================
# HiFT HiFi-GAN Implementation
# ============================================================

class _HiFTGenerator(nn.Module):
    """HiFT: HiFi-GAN with Neural Source Filtering"""
    def __init__(self):
        super().__init__()
        # NSF source (fundamental frequency)
        self.m_source = nn.ModuleDict({
            'l_linear': nn.Linear(9, 1, bias=True)
        })

        # Initial conv: [80, 512, 7]
        self.conv_pre = _ParametrizedConv1d(80, 512, 7, padding=3)

        # Upsample stages
        self.ups = nn.ModuleList([
            _UpBlockParam(512, 256, 8, 16),   # stride=8, kernel=16
            _UpBlockParam(256, 128, 5, 11),   # stride=5, kernel=11
            _UpBlockParam(128, 64, 3, 7),     # stride=3, kernel=7
        ])

        # Multi-period source filters (input=18 channel multi-band, output matches resolution)
        self.source_downs = nn.ModuleList([
            _ParametrizedConv1d(18, 256, 30, padding=15),
            _ParametrizedConv1d(18, 128, 6, padding=3),
            _ParametrizedConv1d(18, 64, 1, padding=0),
        ])

        # Multi-period ResBlocks (source)
        self.source_resblocks = nn.ModuleList([
            _MultiPeriodResBlock(256, 7),
            _MultiPeriodResBlock(128, 7),
            _MultiPeriodResBlock(64, 11),
        ])

        # Main ResBlocks (3 scales × 3 blocks each = 9 total)
        self.resblocks = nn.ModuleList()
        kernel_sizes = [3, 7, 11]
        for scale_idx, channels in enumerate([256, 128, 64]):
            for blk in range(3):
                self.resblocks.append(_MultiPeriodResBlock(channels, kernel_sizes[blk]))

        # Output conv: [64, 18, 7]
        self.conv_post = _ParametrizedConv1d(64, 18, 7, padding=3)

        # F0 predictor
        self.f0_predictor = _F0Predictor()

    def forward(self, x):
        """
        Args:
            x: [B, 80, T] mel-spectrogram
        Returns:
            [B, 18, T'] output (multi-band source coefficients)
        """
        B, C, T_len = x.shape

        # F0 prediction
        f0 = self.f0_predictor(x)  # [B, 1, T]

        # Initial feature: [B, 512, T]
        x = self.conv_pre(x)

        # Stage 0: 256 channels (upsample 8x)
        x = self.ups[0](x)
        x = self.source_resblocks[0](x)
        for i in range(3):
            x = self.resblocks[i](x)

        # Stage 1: 128 channels (upsample 5x)
        x = self.ups[1](x)
        x = self.source_resblocks[1](x)
        for i in range(3):
            x = self.resblocks[3 + i](x)

        # Stage 2: 64 channels (upsample 3x)
        x = self.ups[2](x)
        x = self.source_resblocks[2](x)
        for i in range(3):
            x = self.resblocks[6 + i](x)

        # Output: 18-channel multi-band representation
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x


class _ParametrizedConv1d(nn.Module):
    """Parametrized Conv1d: weight = param0 * param1 (per-channel scale × full kernel)"""
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.param0 = nn.Parameter(torch.randn(out_ch, 1, 1))
        self.param1 = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.padding = padding

    def forward(self, x):
        weight = self.param0 * self.param1
        return F.conv1d(x, weight, self.bias, padding=self.padding)


class _UpBlockParam(nn.Module):
    """Upsample block with parametrized convolution transpose"""
    def __init__(self, in_ch, out_ch, stride, kernel_size):
        super().__init__()
        self.param0 = nn.Parameter(torch.randn(in_ch, 1, 1))
        self.param1 = nn.Parameter(torch.randn(in_ch, out_ch, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.stride = stride
        self.padding = kernel_size // 2
        self.output_padding = stride % 2

    def forward(self, x):
        weight = self.param0 * self.param1
        return F.conv_transpose1d(
            x, weight, self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding
        )


class _MultiPeriodResBlock(nn.Module):
    """Multi-period Residual Block with 3 parallel dilated convs"""
    def __init__(self, channels, kernel_size):
        super().__init__()
        self.convs1 = nn.ModuleList([
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        ])
        self.convs2 = nn.ModuleList([
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            _ParametrizedConv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        ])
        self.activations1 = nn.ModuleList([
            _SnakeActivation(channels) for _ in range(3)
        ])
        self.activations2 = nn.ModuleList([
            _SnakeActivation(channels) for _ in range(3)
        ])

    def forward(self, x):
        residual = x
        for i in range(3):
            x = self.activations1[i](x)
            x = self.convs1[i](x)
            x = self.activations2[i](x)
            x = self.convs2[i](x)
        return x + residual


class _SnakeActivation(nn.Module):
    """Snake activation: x + (1/alpha) * sin(alpha * x)"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        return x + torch.sin(self.alpha * x) / (self.alpha + 1e-9)


class _F0Predictor(nn.Module):
    """F0 (Fundamental Frequency) predictor"""
    def __init__(self):
        super().__init__()
        self.condnet = nn.Sequential(
            _ParametrizedConv1d(80, 512, 3, padding=1),
            _ParametrizedConv1d(512, 512, 3, padding=1),
            _ParametrizedConv1d(512, 512, 3, padding=1),
            _ParametrizedConv1d(512, 512, 3, padding=1),
            _ParametrizedConv1d(512, 512, 3, padding=1),
        )
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        """x: [B, 80, T] → [B, 1, T]"""
        x = self.condnet(x)  # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]
        x = self.classifier(x)  # [B, T, 1]
        return x.transpose(1, 2)  # [B, 1, T]