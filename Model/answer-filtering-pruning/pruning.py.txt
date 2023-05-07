import networkx as nx

class PruningModule():
    def __init__(self, kg, entity_embeddings):
        self.kg = kg
        self.entity_embeddings = entity_embeddings

    def prune(self, topic_entity, candidate_entities):
        filtered_entities = []
        for entity in candidate_entities:
            shortest_path = nx.shortest_path(self.kg, source=topic_entity, target=entity)
            invalid_relations = [r for r in shortest_path if r not in self.kg[topic_entity]]
            if not invalid_relations:
                filtered_entities.append(entity)

        return filtered_entities

    def relation_matching(self, question_embedding, candidate_entities):
        scores = []
        for entity in candidate_entities:
            for relation in self.kg[entity]:
                relation_embedding = self.kg[entity][relation]['embedding']
                relation_similarity = torch.dot(question_embedding, relation_embedding)
                scores.append(relation_similarity)

        # filter out candidate entities with score < 0.5
        filtered_entities = [entity for i, entity in enumerate(candidate_entities) if scores[i] > 0.5]

        return filtered_entities
