import math
import torch
from cache_manager import KVCacheManager


def paged_attention_decode(
    query: torch.Tensor, request_id: str, seq_len: int, cache_manager: KVCacheManager
) -> torch.Tensor:
    """Computes attention for a single incoming query against cached KV vectors.

    Args:
        query: (num_heads, head_dim)
        request_id : active request id
        seq_len : Tokens in context (0 to seq_len - 1)
        cache_manager : KVCacheManager instance

    Returns
        Output : (num_heads, head_dim)

    paged_attention_decode
    1. Read ONLY
        - Assumes K, V for all tokens [0, t] already exist
    2. Gather
        - Looks up block table to find k_vec, v_vec for tokens 0 to t.
    3. Attention method
        - Scores = (q_t @ K_all.T) / sqrt(d_k)
        - Output = Softmax(Scores) @ V_all
    """
    physical_locations = [
        cache_manager.translate_position(request_id, t) for t in range(seq_len)
    ]
    physical_ids = physical_locations[0]
    offsets = physical_locations[1]

    k_all = cache_manager.kv_cache[
        physical_ids, 0, offsets
    ]  # (seq_len, num_heads, head_dim)
    v_all = cache_manager.kv_cache[
        physical_ids, 1, offsets
    ]  # (seq_len, num_heads, head_dim)

    assert k_all.shape == (seq_len, cache_manager.num_heads, cache_manager.head_dim)
    assert v_all.shape == (seq_len, cache_manager.num_heads, cache_manager.head_dim)

    print(k_all)

    # Scores = (q_t @ K_all.T) / sqrt(d_k)
    # (num_heads, 1, head_dim) @ (num_heads, head_dim, seq_len) = (num_heads, 1, seq_len)
    query = query.unsqueeze(dim=0).permute(1, 0, 2)

    assert query.shape == (cache_manager.num_heads, 1, cache_manager.head_dim)

    k_all = k_all.permute(1, 2, 0)
    v_all = v_all.permute(1, 0, 2)

    assert k_all.shape == (cache_manager.num_heads, cache_manager.head_dim, seq_len)
    assert v_all.shape == (cache_manager.num_heads, seq_len, cache_manager.head_dim)

    # (num_heads, 1, head_dim) @ (num_heads, head_dim, seq_len) = (num_heads, 1, seq_len)
    scores = torch.nn.Softmax(dim=-1)(query @ k_all / math.sqrt(cache_manager.head_dim))

    # assert softmax adds upto 1
    assert torch.allclose(torch.ones(cache_manager.num_heads, 1), scores.sum(dim=-1))

    assert scores.shape == (cache_manager.num_heads, 1, seq_len)

    # (num_heads, 1, seq_len) @ (num_heads, seq_len, head_dim) = (num_heads, 1, head_dim)
    outputs = (scores) @ v_all

    outputs = outputs.squeeze(dim=1)

    assert outputs.shape == (cache_manager.num_heads, cache_manager.head_dim)

    return outputs
