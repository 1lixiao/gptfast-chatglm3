from typing import List, Optional

import torch

from model import ChatGLMForConditionalGeneration, Attention, KVCache

class PrefillCacheContext:
    def __init__(self, context: str, context_token_size: int, model: ChatGLMForConditionalGeneration):
        self.context = context
        self.context_token_size = context_token_size
        self.model = model


class PrefillCache:
    def __init__(self, model_device: str, cache_device: str, cache_size: int):
        self.cache = {}
        self.model_device = model_device
        self.cache_device = cache_device
        self.cache_size = cache_size
        self.ctx: Optional[PrefillCacheContext] = None

    def insert(self, key: str, tensors: List[torch.Tensor]):
        assert isinstance(key, str), "Key must be a string"
        if len(self.cache) > self.cache_size:
            del self.cache[next(iter(self.cache))]
        self.cache[key] = tensors

    def get(self, key):
        return self.cache.get(key, None)

    def exists(self, key: str) -> bool:
        return key in self.cache

    def insert_with_chatglm_transformer(self, key: str, model: ChatGLMForConditionalGeneration):
        self.get(key)
        to_save = []
        for i, layer in enumerate(model.transformer.encoder.layers):
            attn: Attention = layer.self_attention
            kv_cache: KVCache = attn.kv_cache
            to_save.append(kv_cache.k_cache.clone().to(self.cache_device))
            to_save.append(kv_cache.v_cache.clone().to(self.cache_device))
        self.insert(key, to_save)

    def load_to_chatglm_transformer(self, key: str, model: ChatGLMForConditionalGeneration):
        lst = self.get(key)
        for i, layer in enumerate(model.transformer.encoder.layers):
            attn: Attention = layer.self_attention
            kv_cache: KVCache = attn.kv_cache
            # kv_cache.register_buffer('k_cache', lst[i * 2].clone().to(self.model_device))
            # kv_cache.register_buffer('v_cache', lst[i * 2 + 1].clone().to(self.model_device))
            # kv_cache.k_cache = lst[i * 2].clone().to(self.model_device)
            # kv_cache.v_cache = lst[i * 2 + 1].clone().to(self.model_device)
            # print(kv_cache.k_cache.shape, lst[i * 2].shape)
            # print(self.ctx.context_token_size)
            kv_cache.k_cache[:, :, 0:self.ctx.context_token_size, :] = lst[i * 2][:, :, 0:self.ctx.context_token_size, :]
            kv_cache.v_cache[:, :, 0:self.ctx.context_token_size, :] = lst[i * 2 + 1][:, :, 0:self.ctx.context_token_size, :]

    def set_context(self, ctx: PrefillCacheContext):
        self.ctx = ctx

    def need_to_prefill(self):
        return self.ctx is not None and not self.exists(self.ctx.context)

    def save(self):
        ctx = self.ctx
        if ctx is None:
            return
        self.insert_with_chatglm_transformer(ctx.context, ctx.model)

    def load(self):
        ctx = self.ctx
        if ctx is None:
            return
        self.load_to_chatglm_transformer(ctx.context, ctx.model)