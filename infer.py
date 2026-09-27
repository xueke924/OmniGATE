#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最简版单条音频推理脚本。

你只需要改 4 个地方：
1. MODEL_MODULE：你的模型文件 import 路径
2. CHECKPOINT_PATH：训练好的 ckpt 路径
3. MIXTURE_PATH：混合音频路径
4. REFERENCE_PATH：注册/reference 音频路径
5. OUT_PATH：保存路径

运行：
python infer_single_tse_simple.py
"""

import sys
import importlib
from pathlib import Path

import torch
import torchaudio
import os


# =========================
# 1. 按你的工程修改这里
# =========================
PROJECT_ROOT = "/home/xueke/real_tse_challenge"

# 例如模型文件是 /root/real_tse/dcfnet_plus/model/mugi_v2_multi.py
# 那这里写 "model.mugi_v2_multi"
MODEL_MODULE = "model.hybrid_mixer_v2_local_global"

# 模型类名
MODEL_CLASS = "OnlineTargetSpeakerExtractionModel"

# ,/home/xueke/dataset/librispeech/libri_2mix_16k/Libri2Mix/wav16k/min/dev/s1/1993-147149-0004_5694-64029-0022.wav,/home/xueke/dataset/librispeech/libri_2mix_16k/Libri2Mix/wav16k/min/dev/s2/1993-147149-0004_5694-64029-0022.wav,/home/xueke/dataset/librispeech/libri_2mix_16k/Libri2Mix/wav16k/min/dev/noise/1993-147149-0004_5694-64029-0022.wav,78400,1,/home/xueke/dataset/librispeech/libri_2mix_16k/Libri2Mix/wav16k/min/dev/s1/1993-147149-0004_5694-64029-0022.wav,1993,1993-147149-0004,147149,1,/home/xueke/dataset/librispeech/libri_2mix_16k/Libri2Mix/wav16k/min/dev/s1_reference_1/1993-147149-0004_5694-64029-0022_1993-147966-0004.wav
# =========================
# 2. 修改你的输入输出路径
# =========================
CHECKPOINT_PATH = "/home/xueke/real_tse_challenge/checkpoint_real_t_local_global_online_speaker_2-4/best-epochepoch=002-valsisdrval_sisdr=6.0961.ckpt"

MIXTURE_PATH = "/home/xueke/dataset/slurp/omni_tse_benchmark_audio/dev/dev_010000/mixture.wav"
REFERENCE_PATH = "/home/xueke/dataset/slurp/omni_tse_benchmark_audio/dev/dev_010000/reference.wav"

OUT_PATH = "/home/xueke/real_tse_challenge/infer_outputs/est.wav"


# =========================
# 3. 基本参数
# =========================
SAMPLE_RATE = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# 训练时 aux_len 如果是 4 秒，这里就保持 4 秒。
# 如果想用完整 reference，把 REF_SECONDS 改成 None。
REF_SECONDS = None


def load_audio_mono(path, sample_rate=16000, device="cuda"):
    wav, sr = torchaudio.load(path)  # [C, T]

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.float()
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    wav = torch.clamp(wav, -1.0, 1.0)

    return wav.unsqueeze(0).to(device)  # [1, 1, T]


def crop_reference_energy(ref, sample_rate=16000, seconds=4.0):
    """
    reference 能量最大片段裁剪。
    ref: [1, 1, T]
    """
    if seconds is None or seconds <= 0:
        return ref

    crop_len = int(round(sample_rate * seconds))
    T = ref.shape[-1]

    if T == crop_len:
        return ref

    if T < crop_len:
        return torch.nn.functional.pad(ref, (0, crop_len - T))

    x = ref[0, 0].float()
    energy = x ** 2

    cumsum = torch.cat(
        [
            torch.zeros(1, device=x.device, dtype=x.dtype),
            torch.cumsum(energy, dim=0),
        ],
        dim=0,
    )

    win_energy = cumsum[crop_len:] - cumsum[:-crop_len]
    start = int(torch.argmax(win_energy).item())

    return ref[..., start:start + crop_len]


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Lightning/DDP 常见前缀处理
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        # 如果 LightningModule 里是 self.model = mugi_block_multi()
        if nk.startswith("model."):
            nk = nk[len("model."):]

        new_state_dict[nk] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    print(f"[Checkpoint] loaded: {ckpt_path}")
    print(f"[Checkpoint] missing keys: {len(missing)}")
    print(f"[Checkpoint] unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("[Checkpoint] first missing:", missing[:5])
    if len(unexpected) > 0:
        print("[Checkpoint] first unexpected:", unexpected[:5])


def save_wav(path, wav, sample_rate=16000):
    """
    wav: [T] or [1, T]
    """
    wav = wav.detach().float().cpu()

    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

    # 防止保存爆音
    peak = wav.abs().max().item()
    if peak > 1.0:
        wav = wav / peak * 0.99

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(str(out_path), wav, sample_rate)
    print(f"[Saved] {out_path}")


def main():
    sys.path.insert(0, PROJECT_ROOT)

    device = torch.device(DEVICE)

    # 1. 实例化模型，使用默认参数
    module = importlib.import_module(MODEL_MODULE)
    ModelClass = getattr(module, MODEL_CLASS)

    model = ModelClass()
    load_checkpoint(model, CHECKPOINT_PATH)

    model = model.to(device)
    model.eval()

    # 2. 读取音频
    mixture = load_audio_mono(MIXTURE_PATH, SAMPLE_RATE, device)
    reference = load_audio_mono(REFERENCE_PATH, SAMPLE_RATE, device)

    original_len = mixture.shape[-1]
    original_ref_len = reference.shape[-1]

    reference = crop_reference_energy(
        reference,
        sample_rate=SAMPLE_RATE,
        seconds=REF_SECONDS,
    )

    print("[Input] mixture:", mixture.shape)
    print("[Input] reference:", reference.shape)

    # 3. 推理
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=False):
            est = model(mixture.float(), reference.float(), original_ref_len)

    print("[Output] raw:", est.shape)

    # 你的模型通常输出 [B, num_source, T]
    if est.dim() == 3:
        est_wav = est[0, 0]
    elif est.dim() == 2:
        est_wav = est[0]
    else:
        raise RuntimeError(f"Unexpected output shape: {est.shape}")

    est_wav = est_wav[:original_len]

    # 4. 保存
    save_wav(OUT_PATH, est_wav, SAMPLE_RATE)


if __name__ == "__main__":
    main()
