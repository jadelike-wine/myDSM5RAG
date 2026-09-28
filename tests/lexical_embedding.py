"""测试用的确定性向量模型 + HuggingFaceEmbedding 替身。

真实项目依赖 sentence-transformers/torch（几百 MB，且要下载模型权重），
不适合放进单元测试和离线诊断脚本。这里提供：

  - `LexicalHashingEmbedding`：把文本按词哈希到固定维度的词袋向量（L2 归一化），
    完全确定性、零依赖，足以让向量通道在测试里"真的按查询排序"
    （llama-index 自带的 MockEmbedding 对所有文本返回同一个常量向量，
    向量通道等于随机排序，锁不住任何检索行为）；
  - `install_fake_huggingface_embedding()`：把
    `llama_index.embeddings.huggingface.HuggingFaceEmbedding` 替换成上述模型的
    兼容壳，供 `DeepSeekRAGSystem._build_embedding_model()` 在测试环境使用。

仅用于测试与诊断脚本，不参与生产推理。
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import types
from typing import List

from llama_index.core.embeddings.mock_embed_model import MockEmbedding

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")


class LexicalHashingEmbedding(MockEmbedding):
    """确定性的词袋哈希向量（仅用于测试）。"""

    def __init__(self, embed_dim: int = 256, **kwargs) -> None:
        super().__init__(embed_dim, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "LexicalHashingEmbedding"

    def _vector_for(self, text: str) -> List[float]:
        dim = int(self.embed_dim)
        vector = [0.0] * dim
        for token in _TOKEN_RE.findall((text or "").lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest()
            vector[int(digest, 16) % dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._vector_for(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._vector_for(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._vector_for(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._vector_for(query)


def install_fake_huggingface_embedding(embed_dim: int = 256) -> type:
    """把 `llama_index.embeddings.huggingface` 换成测试替身（幂等）。

    必须在构造 `DeepSeekRAGSystem` 之前调用：系统的
    `_build_embedding_model()` 在运行时才 import 该模块。
    """

    class _FakeHuggingFaceEmbedding(LexicalHashingEmbedding):
        """签名兼容 HuggingFaceEmbedding，忽略模型名/缓存目录/设备参数。

        BaseEmbedding 是 pydantic 模型，不能塞未声明的属性，因此这些参数只用于
        保持调用签名一致，不保存到实例上。
        """

        def __init__(
            self,
            model_name: str | None = None,
            cache_folder: str | None = None,
            device: str | None = None,
            **kwargs,
        ) -> None:
            del model_name, cache_folder, device
            super().__init__(embed_dim=embed_dim, **kwargs)

        @classmethod
        def class_name(cls) -> str:
            return "FakeHuggingFaceEmbedding"

    module_name = "llama_index.embeddings.huggingface"
    if module_name not in sys.modules or not hasattr(
        sys.modules[module_name], "_is_test_stub"
    ):
        module = types.ModuleType(module_name)
        module.HuggingFaceEmbedding = _FakeHuggingFaceEmbedding
        module._is_test_stub = True
        sys.modules.setdefault("llama_index.embeddings", types.ModuleType("llama_index.embeddings"))
        sys.modules[module_name] = module

    return _FakeHuggingFaceEmbedding
