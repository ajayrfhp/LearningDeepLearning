"""
INFERENCE ENGINE RUNTIME
    1. Compute projections for incoming token t
        - q_t = W_q(x_t)
        - k_t = W_k(k_t)
        - v_t = W_v(v_t)
    2. Write / Allocate step
        - cache_manager.store_token_kv(request_id, k_t, v_t)

Inference engine passes (q_t, request_id, seq_len = t + 1) to (paged_attention_decode)

paged_attention_decode
    1. Read ONLY
        - Assumes K, V for all tokens [0, t] already exist
    2. Gather
        - Looks up block table to find k_vec, v_vec for tokens 0 to t.
    3. Attention method
        - Scores = (q_t @ K_all.T) / sqrt(d_k)
        - Output = Softmax(Scores) @ V_all

"""

from cache_manager import KVCacheManager
import torch
from paged_attention import paged_attention_decode

num_blocks = 3
block_size = 4
num_heads = 10
head_dim = 16

kv_cache_manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)


ones = torch.ones((num_heads, head_dim))
kv_cache_manager.store_token_kv("req_0", 0, ones * 0, ones * 0)
kv_cache_manager.store_token_kv("req_0", 1, ones * 1, ones * 1)

decoded_tokens, _ = paged_attention_decode(ones * 2, "req_0", 2, kv_cache_manager)

print(decoded_tokens.shape)
print(decoded_tokens)
