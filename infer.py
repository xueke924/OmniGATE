
import sys
import importlib
from pathlib import Path

import torch
import torchaudio
import os


PROJECT_ROOT = "/home/xueke/real_tse_challenge"

# /root/real_tse/dcfnet_plus/model/mugi_v2_multi.py
# "model.mugi_v2_multi"
MODEL_MODULE = "model.hybrid_mixer_v2_local_global"

# class name
MODEL_CLASS = "OnlineTargetSpeakerExtractionModel"

CHECKPOINT_PATH = "/home/xueke/real_tse_challenge/checkpoint_real_t_local_global_online_speaker_2-4/best-epochepoch=002-valsisdrval_sisdr=6.0961.ckpt"

MIXTURE_PATH = "/home/xueke/dataset/slurp/omni_tse_benchmark_audio/dev/dev_010000/mixture.wav"
REFERENCE_PATH = "/home/xueke/dataset/slurp/omni_tse_benchmark_audio/dev/dev_010000/reference.wav"

OUT_PATH = "/home/xueke/real_tse_challenge/infer_outputs/est.wav"

SAMPLE_RATE = 16000
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    module = importlib.import_module(MODEL_MODULE)
    ModelClass = getattr(module, MODEL_CLASS)

    model = ModelClass()
    load_checkpoint(model, CHECKPOINT_PATH)

    model = model.to(device)
    model.eval()

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

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=False):
            est = model(mixture.float(), reference.float(), original_ref_len)

    print("[Output] raw:", est.shape)

    if est.dim() == 3:
        est_wav = est[0, 0]
    elif est.dim() == 2:
        est_wav = est[0]
    else:
        raise RuntimeError(f"Unexpected output shape: {est.shape}")

    est_wav = est_wav[:original_len]

    save_wav(OUT_PATH, est_wav, SAMPLE_RATE)


if __name__ == "__main__":
    main()
