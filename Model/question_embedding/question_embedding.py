import torch
import torch.nn as nn
from transformers import RobertaModel


class QuestionEmbedding(nn.Module):
    def __init__(self, num_entities, embedding_dim):
        super(QuestionEmbedding, self).__init__()

        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.complex_dim = embedding_dim

    def forward(self, question, topic_entity, candidate_entities):
        # Embed question using RoBERTa
        question_embedding = self.roberta(question).last_hidden_state[:, 0, :]

        # Project question embedding to complex space
        question_embedding = self.fc(question_embedding)

        # Get topic entity embedding
        topic_entity_embedding = self.entity_embeddings(topic_entity)

        # Get candidate entity embeddings
        candidate_entity_embeddings = self.entity_embeddings(candidate_entities)

        # Compute ComplEx scores
        scores = torch.sum(
            topic_entity_embedding * question_embedding * candidate_entity_embeddings,
            dim=-1
        )

        return scores
