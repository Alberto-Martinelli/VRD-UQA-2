import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class QuestionClassifier:
    def __init__(self, categories, bert_embedding):
        self.categories = categories
        self.bert_embedding = bert_embedding
        self.category_embeddings = self._generate_category_embeddings()

    def _generate_category_embeddings(self):
        return {cat: np.array([self.bert_embedding.get_embedding(q) for q in examples]) 
                for cat, examples in self.categories.items()}

    def classify_question(self, question, similarity_threshold=0.7):
        question_embedding = self.bert_embedding.get_embedding(question)
        similarities = {cat: cosine_similarity([question_embedding], emb).mean() 
                        for cat, emb in self.category_embeddings.items()}
        max_similarity = max(similarities.values())
        if max_similarity >= similarity_threshold:
            return list(self.categories.keys()).index(max(similarities, key=similarities.get))
        else:
            return -1