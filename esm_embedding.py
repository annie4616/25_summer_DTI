import torch
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# pip install fair-esm  (또는 esm==2.x 패키지)
import esm

@torch.no_grad()
def load_esm2_650m(device: str = 'cuda'):
    # 33layers

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

@torch.no_grad()
def esm_embedding(
    sequences: List[str],
    model,
    alphabet,
    batch_converter,
    device: str = "cuda",
    repr_layer: int = 33,
    # amp: bool = False,
) -> torch.Tensor:
    # 시퀀스 리스트를 받아 ESM2 per-residue representation을 얻고,
    # padding, BOS, EOS를 제외한 residue에 대해 mean pooling한 [B,D] 텐서를 반환
    # ESM batch 입력 준비
    batch = [("",s) for s in sequences]
    _, _, tokens = batch_converter(batch)
    tokens = tokens.to(device)

    # AMP 옵션 (원하면 켜기)
    # autocast_ctx = torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda")))
    # with autocast_ctx:
    #     out = model(tokens, repr_layers=[repr_layer], return_contacts=False)
    #     reps = out["representations"][repr_layer]  # [B, L, D]

    # mask 만들기
    pad_idx = alphabet.padding_idx
    bos_idx = alphabet.bos_idx
    eos_idx = alphabet.eos_idx

    # 유효토큰: not padding
    valid = tokens != pad_idx
    valid = valid & (tokens != bos_idx) & (tokens != eos_idx)

    # 모든 토큰이 특수 토큰인 경우 mean이 NaN이 될 수 있음(길이가 0)
    valid_lens = valid.sum(dim=1).clamp_min(1)

    valid = valid.unsqueeze(-1)  # [B, L, 1]
    reps = reps*valid  # [B, L, D] (masked)
    pooled = reps.sum(dim=1)/valid_lens.unsqueeze(-1) #[B, D]

    return pooled.to("cpu", dtype=torch.float32)  # CPU로 옮기고 float32로 변환