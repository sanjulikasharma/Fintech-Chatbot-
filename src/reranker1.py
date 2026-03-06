from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        # Load cross-encoder reranker model
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query, docs):

        if not docs:
            return docs

        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(scores, docs))

        # sort by relevance score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # return top 3
        return [doc for _, doc in scored_docs[:3]]