from cache_manager import KVCacheManager

num_blocks = 3
block_size = 4
num_heads = 10
head_dim = 16

kv_cache_manager = KVCacheManager(num_blocks, block_size, num_heads, head_dim)

# request_id 0, store 10 tokens.
kv_cache_manager.allocate(0, 8)

print(kv_cache_manager.block_tables)
print(kv_cache_manager.free_blocks)

print(kv_cache_manager.translate_position(0, 1))
