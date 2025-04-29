import numpy as np
from openai import OpenAI
from topics import get_sample_topics

class EmbeddingStorage:
    def __init__(self, openai: OpenAI):
        self.embeddings = []  
        self.openai = openai
        topics = get_sample_topics()
        print(f"initializing embeddings for {len(topics)} topics")
        self.initialize_embeddings(topics=topics)

    def add_embedding(self, embedding: list[float]):
        self.embeddings.append(np.array(embedding))
        
    def initialize_embeddings(self, topics: list):
        for topic in topics:
            response = self.openai.embeddings.create(input=topic["description"], model="text-embedding-ada-002")
            embedding = response.data[0].embedding  # Correct way to access embedding
            self.embeddings.append(np.array(embedding))
            
    def get_embeddings(self):
        return np.array(self.embeddings)