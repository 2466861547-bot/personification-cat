"""
宠物知识检索器: 基于向量数据库的 RAG 检索
使用 ChromaDB + sentence-transformers 实现语义检索

支持双模型策略:
- Mac MPS/CPU: 自动使用轻量 Embedding 模型 (bge-small-zh)
- GPU (CUDA): 自动使用高质量 Embedding 模型 (bge-large-zh)
"""

import os
import socket
from typing import List, Dict, Optional
from dataclasses import dataclass

from .knowledge_base import PetKnowledgeBase


# ============ Embedding 配置 ============

@dataclass
class EmbeddingConfig:
    """Embedding 模型配置 - 支持硬件自适应"""
    # 轻量模型 (MPS/CPU 用，约 23MB)
    lightweight_model: str = "BAAI/bge-small-zh-v1.5"
    # 高质量模型 (GPU 用，约 1.3GB)
    high_quality_model: str = "BAAI/bge-large-zh-v1.5"
    # 向量维度 (不同模型维度不同，自动检测)
    dimension: int = 0

    @classmethod
    def auto(cls) -> "EmbeddingConfig":
        """根据硬件自动选择配置"""
        device = _detect_device()
        if device == "cuda":
            print("🤖 检测到 GPU，使用高质量 Embedding 模型")
            return cls()  # 默认就是 high_quality
        else:
            model = cls()
            # Mac MPS / CPU: 使用轻量模型
            model.high_quality_model = model.lightweight_model
            print("🍎 检测到 Mac/CPU，使用轻量 Embedding 模型: " + model.lightweight_model)
            return model


def _detect_device() -> str:
    """检测设备类型"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _check_network(timeout: int = 3) -> bool:
    """检测网络是否可用 - 同时检查 HuggingFace 和国内镜像"""
    # 优先检测国内镜像 (更稳定)
    for host, port in [("hf-mirror.com", 443), ("huggingface.co", 443)]:
        try:
            socket.setdefaulttimeout(timeout)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            continue
    return False


# ============ 向量检索器 ============

class PetRetriever:
    """宠物知识向量检索器"""

    def __init__(
        self,
        knowledge_base: PetKnowledgeBase,
        embedding_config: Optional[EmbeddingConfig] = None,
        collection_name: str = "pet_knowledge",
        persist_dir: str = "./data/chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.knowledge_base = knowledge_base
        self.embedding_config = embedding_config or EmbeddingConfig.auto()
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._embed_model = None
        self._collection = None
        self._chroma_client = None
        self._is_embedding_ready = False
        self._embedding_error = None

    @property
    def embedding_model_name(self) -> str:
        """当前使用的 embedding 模型名"""
        return self.embedding_config.high_quality_model

    def _init_embeddings(self) -> bool:
        """初始化 embedding 模型 - 三级降级策略"""
        if self._embed_model is not None:
            return True

        # 设置 HuggingFace 镜像源 (国内网络加速)
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        from sentence_transformers import SentenceTransformer

        models_to_try = []
        # 1. 优先尝试配置的主模型
        models_to_try.append(self.embedding_config.high_quality_model)
        # 2. 降级到轻量模型 (如果不同于主模型)
        if self.embedding_config.lightweight_model != self.embedding_config.high_quality_model:
            models_to_try.append(self.embedding_config.lightweight_model)

        network_ok = _check_network()
        print(f"加载 Embedding 模型 (网络: {'✅ 已连接' if network_ok else '❌ 未连接'})")

        last_error = None
        for model_name in models_to_try:
            # 步骤 1: 尝试离线加载
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                print(f"  尝试离线加载: {model_name}")
                self._embed_model = SentenceTransformer(model_name)
                print(f"  ✅ 离线加载成功: {model_name}")
                self._embedding_config = model_name
                self._is_embedding_ready = True
                return True
            except Exception as e:
                last_error = e
                self._embed_model = None
                print(f"  ⚠️  本地无缓存 {model_name}")

            # 步骤 2: 如果有网络，尝试在线下载
            if network_ok:
                try:
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    print(f"  在线下载: {model_name}")
                    self._embed_model = SentenceTransformer(model_name)
                    print(f"  ✅ 加载成功: {model_name}")
                    self._embedding_config = model_name
                    self._is_embedding_ready = True
                    return True
                except Exception as e:
                    last_error = e
                    self._embed_model = None
                    print(f"  ⚠️  在线加载 {model_name} 失败")

            # 步骤 3: 尝试降级到下一个模型 (如果有不同的模型)
            if model_name != models_to_try[-1]:
                print(f"  🔻 尝试下一个模型...")
            else:
                break

        # 所有模型都失败
        self._embedding_error = last_error
        self._is_embedding_ready = False
        print(f"\n❌ Embedding 模型加载失败!")
        print(f"💡 解决方法:")
        if self.embedding_config.lightweight_model != self.embedding_config.high_quality_model:
            print(f"   1. 手动下载轻量模型: huggingface-cli download {self.embedding_config.lightweight_model}")
            print(f"   2. 手动下载高质量模型: huggingface-cli download {self.embedding_config.high_quality_model}")
        else:
            print(f"   1. 手动下载模型: huggingface-cli download {self.embedding_config.lightweight_model}")
        print(f"   3. 使用国内镜像加速: HF_ENDPOINT=https://hf-mirror.com huggingface-cli download <模型名>")
        print(f"   4. 下载后复制到 ~/.cache/huggingface/hub/ 目录")
        print(f"   5. 临时跳过 RAG: python scripts/inference.py --no-rag")
        return False

    def _init_chroma(self):
        """初始化 ChromaDB"""
        if self._chroma_client is None:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "宠物声音与行为知识库"},
            )

    def build_index(self) -> bool:
        """构建向量索引 - 返回是否成功，打印详细日志"""
        if not self._init_embeddings():
            return False

        self._init_chroma()
        documents = self.knowledge_base.to_documents()

        # 打印知识库详情
        self._print_kb_summary()

        # 如果已有数据,先清空
        if self._collection.count() > 0:
            existing = self._collection.get()
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])

        # 批量插入
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(texts))]

        # 生成 embedding
        print(f"\n📊 开始向量化: {len(texts)} 条文档")
        embeddings = self._embed_model.encode(texts, show_progress_bar=True)
        embeddings_list = embeddings.tolist()

        self._collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        model_name = getattr(self, '_embedding_config', 'unknown')
        print(f"\n✅ 知识库索引构建完成")
        print(f"   文档数: {len(texts)} 条")
        print(f"   向量维度: {len(embeddings_list[0]) if embeddings_list else 0} 维")
        print(f"   Embedding 模型: {model_name}")
        print(f"   存储位置: {self.persist_dir}")

        # 打印存储目录结构
        self._print_chroma_structure()

        return True

    def _print_kb_summary(self):
        """打印知识库内容摘要"""
        entries = self.knowledge_base.entries
        from collections import Counter

        pet_types = Counter(e.pet_type for e in entries)
        emotions = Counter(e.emotion for e in entries)
        breeds = Counter(e.breed for e in entries)

        print(f"\n📚 知识库内容摘要")
        print(f"{'='*50}")
        print(f"  总条目数: {len(entries)} 条")
        print(f"  宠物类型: {dict(pet_types)}")
        print(f"  情绪分布:")
        for emo, count in emotions.most_common():
            print(f"    - {emo}: {count} 条")
        print(f"  品种分布 ({len(breeds)} 个品种):")
        for breed, count in breeds.most_common(10):
            print(f"    - {breed}: {count} 条")
        if len(breeds) > 10:
            print(f"    ... 还有 {len(breeds) - 10} 个品种")

        # 打印前3条样例
        print(f"\n  📖 知识条目示例 (前3条):")
        for i, entry in enumerate(entries[:3], 1):
            print(f"    [{i}] {entry.pet_type}/{entry.breed}/{entry.emotion}")
            print(f"        声音: {entry.sound_description[:60]}...")
            print(f"        含义: {entry.meaning}")
            print(f"        建议: {entry.advice[:60]}...")
            print()

    def _print_chroma_structure(self):
        """打印 ChromaDB 存储目录结构"""
        if not os.path.exists(self.persist_dir):
            return
        files = []
        for root, dirs, filenames in os.walk(self.persist_dir):
            for fn in filenames:
                fp = os.path.join(root, fn)
                size = os.path.getsize(fp)
                files.append((os.path.relpath(fp, self.persist_dir), size))

        if files:
            print(f"\n📁 ChromaDB 存储结构 ({self.persist_dir}):")
            for path, size in sorted(files):
                if size > 1024:
                    print(f"    {path} ({size/1024:.1f} KB)")
                else:
                    print(f"    {path} ({size} bytes)")
            total_size = sum(s for _, s in files)
            print(f"    总计: {len(files)} 个文件, {total_size/1024:.1f} KB")

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """向量检索"""
        if not self._is_embedding_ready:
            return []
        if self._collection is None:
            self._init_embeddings()
            self._init_chroma()
        if not self._is_embedding_ready:
            return []

        top_k = top_k or self.top_k

        # 生成查询向量
        query_embedding = self._embed_model.encode([query])[0].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # 格式化结果
        search_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                search_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],
                })

        return search_results

    def search_by_emotion(
        self,
        pet_type: str,
        emotion: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """按情绪检索"""
        query = f"{pet_type} {emotion} 叫声 含义 建议"
        results = self.search(query, top_k=top_k)

        # 过滤匹配的宠物类型
        filtered = [
            r for r in results
            if r["metadata"].get("pet_type") == pet_type
            or r["metadata"].get("pet_type") == "通用"
        ]
        return filtered

    def get_context_for_llm(
        self,
        query: str,
        top_k: int = None,
    ) -> str:
        """获取 LLM 上下文 - 返回格式化的检索结果"""
        if not self._is_embedding_ready:
            return ""
        results = self.search(query, top_k=top_k)

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"【知识{i}】\n{r['text']}\n相关度: {r['score']:.2f}\n"
            )

        return "\n".join(context_parts)

    def search_with_details(
        self,
        query: str,
        top_k: int = None,
    ) -> Dict:
        """带详细日志的检索 - 返回结果和统计信息"""
        if not self._is_embedding_ready:
            return {"results": [], "total": 0, "message": "Embedding 模型未就绪"}

        results = self.search(query, top_k=top_k)
        return {
            "results": results,
            "total": len(results),
            "query": query,
            "top_k": top_k or self.top_k,
        }

    def add_knowledge(self, text: str, metadata: Dict = None):
        """增量添加知识"""
        if not self._is_embedding_ready:
            if not self._init_embeddings():
                return
        if self._collection is None:
            self._init_chroma()

        embedding = self._embed_model.encode([text])[0].tolist()
        doc_id = f"doc_custom_{self._collection.count()}"

        self._collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        print(f"已添加知识: {doc_id}")

    @property
    def is_ready(self) -> bool:
        """检查 embedding 是否就绪"""
        return self._is_embedding_ready
