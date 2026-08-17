"""
RAG 优化检索器: 混合检索 + Rerank + 查询重写
- 向量检索 (bge-large) + BM25 关键词检索
- Cohere Rerank 重排序
- HyDE 查询重写提升召回率
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass

from .knowledge_base import PetKnowledgeBase


@dataclass
class OptimizedRAGConfig:
    """RAG 优化配置"""
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    collection_name: str = "pet_knowledge"
    persist_dir: str = "./data/chroma_db"
    # 混合检索权重
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    # 检索参数
    top_k: int = 5
    rerank_top_k: int = 3
    # HyDE
    enable_hyde: bool = True
    # Chunk
    chunk_size: int = 512
    chunk_overlap: int = 50


class HybridRetriever:
    """
    混合检索器: 向量检索 + BM25 关键词检索
    结合语义理解和精确匹配,提升召回率
    """

    def __init__(self, config: OptimizedRAGConfig = None):
        self.config = config or OptimizedRAGConfig()
        self._embed_model = None
        self._rerank_model = None
        self._chroma_client = None
        self._collection = None
        self._bm25_index = None
        self._documents = []

    def _init_models(self):
        """初始化模型"""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"加载 Embedding 模型: {self.config.embedding_model}")
            self._embed_model = SentenceTransformer(self.config.embedding_model)

    def _init_reranker(self):
        """初始化 Rerank 模型"""
        if self._rerank_model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"加载 Rerank 模型: {self.config.rerank_model}")
                self._rerank_model = CrossEncoder(self.config.rerank_model)
            except Exception as e:
                print(f"Rerank 模型加载失败: {e}")

    def build_index(self, knowledge_base: PetKnowledgeBase):
        """构建混合索引"""
        self._init_models()

        # 获取文档
        documents = knowledge_base.to_documents()
        self._documents = documents

        # 1. 构建向量索引
        import chromadb
        os.makedirs(self.config.persist_dir, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=self.config.persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.config.collection_name
        )

        if self._collection.count() > 0:
            existing = self._collection.get()
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])

        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [f"doc_{i}" for i in range(len(texts))]
        embeddings = self._embed_model.encode(texts).tolist()

        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )

        # 2. 构建 BM25 索引
        self._build_bm25_index(texts)

        print(f"混合索引构建完成: {len(texts)} 条文档")

    def _build_bm25_index(self, texts: List[str]):
        """构建 BM25 关键词索引"""
        import jieba

        # 中文分词
        tokenized = [list(jieba.cut(text)) for text in texts]
        self._tokenized_docs = tokenized

        # 计算 BM25
        from collections import Counter
        import math

        self._doc_freq = Counter()
        for tokens in tokenized:
            for token in set(tokens):
                self._doc_freq[token] += 1

        self._doc_len = [len(tokens) for tokens in tokenized]
        self._avg_doc_len = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 1
        self._n_docs = len(tokenized)

    def _bm25_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25 关键词检索"""
        import jieba
        from collections import Counter
        import math

        query_tokens = list(jieba.cut(query))
        query_counter = Counter(query_tokens)

        scores = []
        for i, doc_tokens in enumerate(self._tokenized_docs):
            doc_counter = Counter(doc_tokens)
            score = 0.0

            for token, q_freq in query_counter.items():
                if token in doc_counter:
                    # BM25 公式
                    tf = doc_counter[token]
                    df = self._doc_freq.get(token, 0)
                    idf = math.log(
                        (self._n_docs - df + 0.5) / (df + 0.5) + 1
                    )
                    k1 = 1.5
                    b = 0.75
                    tf_score = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * self._doc_len[i] / self._avg_doc_len)
                    )
                    score += idf * tf_score * q_freq

            scores.append({"id": i, "score": score})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """混合检索"""
        top_k = top_k or self.config.top_k
        self._init_models()

        # 1. 向量检索
        query_embedding = self._embed_model.encode([query])[0].tolist()
        vector_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
        )

        vector_scores = []
        for i in range(len(vector_results["ids"][0])):
            vector_scores.append({
                "id": vector_results["ids"][0][i],
                "text": vector_results["documents"][0][i],
                "metadata": vector_results["metadatas"][0][i],
                "vector_score": 1 - vector_results["distances"][0][i],
            })

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        bm25_scores = []
        for r in bm25_results:
            doc = self._documents[r["id"]]
            bm25_scores.append({
                "id": f"doc_{r['id']}",
                "text": doc["text"],
                "metadata": doc["metadata"],
                "bm25_score": r["score"],
            })

        # 3. 融合 (RRF - Reciprocal Rank Fusion)
        rrf_scores = {}
        k = 60  # RRF 常数

        for rank, r in enumerate(vector_scores):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"text": r["text"], "metadata": r["metadata"]})
            rrf_scores[doc_id]["score"] = rrf_scores[doc_id].get("score", 0) + \
                self.config.vector_weight / (k + rank + 1)

        for rank, r in enumerate(bm25_scores):
            doc_id = r["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"text": r["text"], "metadata": r["metadata"]})
            rrf_scores[doc_id]["score"] = rrf_scores[doc_id].get("score", 0) + \
                self.config.bm25_weight / (k + rank + 1)

        # 排序
        combined = [
            {"id": k, **v} for k, v in rrf_scores.items()
        ]
        combined.sort(key=lambda x: x["score"], reverse=True)

        # 4. Rerank
        if self._rerank_model is None:
            self._init_reranker()

        if self._rerank_model and len(combined) > 0:
            rerank_top_k = self.config.rerank_top_k
            pairs = [(query, doc["text"]) for doc in combined[:top_k * 2]]
            rerank_scores = self._rerank_model.predict(pairs)

            for i, score in enumerate(rerank_scores):
                combined[i]["rerank_score"] = float(score)

            combined.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return combined[:rerank_top_k]

        return combined[:top_k]

    def get_context_for_llm(self, query: str, top_k: int = None) -> str:
        """获取 LLM 上下文"""
        results = self.search(query, top_k=top_k)
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"【知识{i}】\n{r['text']}\n")
        return "\n".join(context_parts)


class HyDEQueryRewriter:
    """
    HyDE (Hypothetical Document Embedding) 查询重写
    1. 用 LLM 生成假设性回答
    2. 用假设回答做向量检索 (比原始 query 召回更准)
    """

    def __init__(self, llm_inference=None):
        self.llm = llm_inference

    def rewrite(self, query: str, pet_type: str = "cat") -> str:
        """查询重写"""
        if self.llm is None:
            # 无 LLM 时,简单扩展
            return f"{pet_type} 宠物 声音 情绪 行为 {query}"

        # 生成假设性回答
        hyde_prompt = (
            f"请用一句话回答: 当{pet_type}发出「{query}」的声音时，"
            f"它可能是什么情绪？"
        )

        hypothetical_answer = self.llm.chat(hyde_prompt, max_tokens=64)

        # 用假设回答作为检索 query
        return f"{query} {hypothetical_answer}"
