import torch
from collections import defaultdict


class KVCacheManager:
    def __init__(self, num_blocks, block_size, num_heads, head_dim):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_cache = torch.zeros((num_blocks, 2, block_size, num_heads, head_dim))
        self.free_blocks = torch.ones(num_blocks)
        self.block_tables = defaultdict(list)  # dictionary of active blocks per request

    def allocate(self, request_id, num_tokens):
        num_blocks_assigned = len(self.block_tables[request_id])
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if num_blocks_assigned >= num_blocks_needed:
            return True
        for i in range(self.free_blocks.shape[0]):
            if self.free_blocks[i]:
                # physical block is free, grab it
                self.free_blocks[i] = 0
                self.block_tables[request_id].append(i)
                num_blocks_assigned += 1
            if num_blocks_assigned >= num_blocks_needed:
                return True
        return False

    def translate_position(self, request_id, token_idx):
        """Given a token position index, return (physical_block_id, offset)"""
        logical_block_id = token_idx // self.block_size
        if logical_block_id >= len(self.block_tables[request_id]):
            self.allocate(request_id, token_idx + 1)
        physical_block_id = self.block_tables[request_id][logical_block_id]
        offset = token_idx % self.block_size
        return physical_block_id, offset

    def store_token_kv(self, request_id, token_idx, k_vec, v_vec):
        physical_block, offset = self.translate_position(request_id, token_idx)
        self.kv_cache[physical_block, 0, offset] = k_vec
        self.kv_cache[physical_block, 1, offset] = v_vec
