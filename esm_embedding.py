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

@torch.no_grad()
def build_protein_cache_esm2_650m(
    protein_ids: List[str],
    protein_seqs: List[str],
    out_path: str,
    batch_size: int = 16,
    device: str = "cuda",
    amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    ESM2 650M으로 mean pooled 임베딩을 추출하여 {id: [D]} 딕셔너리로 저장.
    """
    assert len(protein_ids) == len(protein_seqs)
    model, alphabet, batch_converter = load_esm2_650m(device=device)

    cache: Dict[str, torch.Tensor] = {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(0, len(protein_seqs), batch_size), desc="Embedding (ESM2 650M)"):
        batch_ids  = protein_ids[i:i+batch_size]
        batch_seqs = protein_seqs[i:i+batch_size]

        pooled = esm2_embed_mean(
            batch_seqs,
            model=model,
            alphabet=alphabet,
            batch_converter=batch_converter,
            device=device,
            repr_layer=33,
            amp=amp,
        )  # [B, 1280]

        for pid, vec in zip(batch_ids, pooled):
            cache[pid] = vec.contiguous()  # [1280], float32, cpu

    torch.save(cache, out_path)
    return cache


if __name__ == "__main__":
    # ==== 사용 예시 ====
    # 여러분의 실제 데이터로 교체하세요.
    # protein_ids와 protein_seqs의 순서는 반드시 1:1 매핑이어야 합니다.
    protein_ids = ["P001", "P002", "P003"]
    protein_seqs = [
        "MKWVTFISLLFLFSSAYS",
        "ACDEFGHIKLMNPQRSTVWY",
        "SSSSSSSSSSSSSSSSSS"
    ]

    cache_path = "cache/protein_embeds_esm2_650m_mean.pt"
    build_protein_cache_esm2_650m(
        protein_ids=protein_ids,
        protein_seqs=protein_seqs,
        out_path=cache_path,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=True,  # GPU 사용 시 True 권장
    )
    print(f"Saved: {cache_path}")