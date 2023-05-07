import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplExEmbedding(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super().__init__()
        self.entity_embeddings_real = nn.Embedding(num_entities, embedding_dim)
        self.entity_embeddings_img = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings_real = nn.Embedding(num_relations, embedding_dim)
        self.relation_embeddings_img = nn.Embedding(num_relations, embedding_dim)

    def forward(self, h, r, t):
        real_score = torch.sum(self.entity_embeddings_real(h) * self.relation_embeddings_real(r) * self.entity_embeddings_real(t) +
                               self.entity_embeddings_real(h) * self.relation_embeddings_img(r) * self.entity_embeddings_img(t) +
                               self.entity_embeddings_img(h) * self.relation_embeddings_real(r) * self.entity_embeddings_img(t) -
                               self.entity_embeddings_img(h) * self.relation_embeddings_img(r) * self.entity_embeddings_real(t), dim=1)
        img_score = torch.sum(self.entity_embeddings_img(h) * self.relation_embeddings_real(r) * self.entity_embeddings_real(t) -
                              self.entity_embeddings_img(h) * self.relation_embeddings_img(r) * self.entity_embeddings_img(t) +
                              self.entity_embeddings_real(h) * self.relation_embeddings_real(r) * self.entity_embeddings_img(t) +
                              self.entity_embeddings_real(h) * self.relation_embeddings_img(r) * self.entity_embeddings_real(t), dim=1)
        score = torch.stack([real_score, img_score], dim=1)
        return score
