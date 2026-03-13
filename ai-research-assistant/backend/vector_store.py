import numpy as np

class VectorStore:

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, embeddings, chunks):
        for i in range(len(embeddings)):
            self.vectors.append(embeddings[i])
            self.texts.append(chunks[i])

    def search(self, query_embedding, top_k=3):

        scores = []

        for vector in self.vectors:
            score = np.dot(query_embedding, vector)
            scores.append(score)

        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i],
                             reverse=True)[:top_k]

        results = [self.texts[i] for i in top_indices]

        return results