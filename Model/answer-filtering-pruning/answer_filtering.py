import torch.nn as nn

class AnswerFilteringModule(nn.Module):
    def __init__(self, entity_embeddings):
        super().__init__()
        self.entity_embeddings = entity_embeddings

    def forward(self, head_entity, question_embedding):
        scores = []
        for entity_embedding in self.entity_embeddings:
            score = self.complex_scoring_function(head_entity, question_embedding, entity_embedding)
            scores.append(score)

        # return the entity with the highest score
        best_entity_idx = torch.argmax(torch.stack(scores))
        best_entity_embedding = self.entity_embeddings[best_entity_idx]

        return best_entity_embedding

    def complex_scoring_function(self, head_entity, question_embedding, answer_entity):
        # Implement the ComplEx scoring function here
        # ComplEx scoring function is f(eh, eq, ea) = Re(eh*conj(eq)*ea)
        return score

