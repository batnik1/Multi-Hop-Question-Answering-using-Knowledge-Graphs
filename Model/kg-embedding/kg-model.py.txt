import torch
import torch.nn as nn
from kg_embeddings_utils import ComplExEmbedding

class KGEmbeddingModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_embeddings = ComplExEmbedding(num_entities, num_relations, embedding_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, h, r, t):
        score = self.entity_embeddings(h, r, t)
        score = self.dropout(score)
        return score
