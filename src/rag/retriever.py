"""
宠物知识检索器: 基于向量数据库的 RAG 检索
使用 ChromaDB + sentence-transformers 实现语义检索
"""

import os
from typing import List, Dict, Optional

from .knowledge_base import PetKnowledgeBase


class PetRetriever:
    """宠物知识向量检索器"""

    def __init__(
        self,
        knowledge_base: PetKnowledgeBase,
        embedding_model: str = "BAAI/bge-large-zh-v1.5",
        collection_name: str = "pet_knowledge",
        persist_dir: str = "./data/chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ):
        self.knowledge_base = knowledge_base
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._embed_model = None
        self._collection = None
        self._chroma_client = None

    def _init_embeddings(self):
        """初始化 embedding 模型"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"加载 Embedding 模型: {self.embedding_model_name}")
            self._embed_model = SentenceTransformer(self.embedding_model_name)

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

    def build_index(self):
        """构建向量索引"""
        """构建向量索引"""
        """构建向量索引"""
        """构建向量索引"""
        self._init_embeddings()
        self._init_chroma()

        documents = self.knowledge_base.to_documents()

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
        embeddings = self._embed_model.encode(texts, show_progress_bar=True)
        embeddings_list = embeddings.tolist()

        self._collection.add(
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        print(f"知识库索引构建完成: {len(texts)} 条文档")

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """向量检索"""
        if self._collection is None:
            self._init_embeddings()
            self._init_chroma()

        top_k = top_k or self.top_k

        # 生成查询向量
        query_embedding = self._embed_model.encode([query])[0].tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # 格式化结果
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # 距离转相似度
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
        """获取 LLM 上下文"""
        results = self.search(query, top_k=top_k)

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"【知识{i}】\n{r['text']}\n相关度: {r['score']:.2f}\n"
            )

        return "\n".join(context_parts)

    def add_knowledge(self, text: str, metadata: Dict = None):
        """增量添加知识"""
        if self._collection is None:
            self._init_embeddings()
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
