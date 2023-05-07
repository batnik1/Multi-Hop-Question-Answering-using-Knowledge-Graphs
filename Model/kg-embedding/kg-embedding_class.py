import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class KGEmbeddingModule(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(KGEmbeddingModule, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim, padding_idx=None)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim, padding_idx=None)
        self.entity_embeddings.weight.data.normal_(0, 1.0 / np.sqrt(num_entities))
        self.relation_embeddings.weight.data.normal_(0, 1.0 / np.sqrt(num_relations))

    def forward(self, heads, tails, relations):
        head_embeddings = self.entity_embeddings(heads)
        tail_embeddings = self.entity_embeddings(tails)
        relation_embeddings = self.relation_embeddings(relations)
        complex_head_embeddings = head_embeddings[:, 0:self.embedding_dim] + 1j * head_embeddings[:, self.embedding_dim:]
        complex_tail_embeddings = tail_embeddings[:, 0:self.embedding_dim] + 1j * tail_embeddings[:, self.embedding_dim:]
        complex_relation_embeddings = relation_embeddings[:, 0:self.embedding_dim] + 1j * relation_embeddings[:, self.embedding_dim:]
        scores = (complex_head_embeddings * complex_relation_embeddings.conj() * complex_tail_embeddings).sum(dim=1)
        return scores
