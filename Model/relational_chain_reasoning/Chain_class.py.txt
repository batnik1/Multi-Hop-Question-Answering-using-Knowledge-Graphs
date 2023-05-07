import numpy as np
from scipy.spatial.distance import cdist

class RelationChainReasoning:
    def __init__(self, kg_embedding_module, kg, max_iterations=3, k=10):
        self.kg_embedding_module = kg_embedding_module
        self.kg = kg
        self.max_iterations = max_iterations
        self.k = k
        
    def run(self, topic_entity, top_k_entities):
        result = self.expand_neighbors(topic_entity, top_k_entities)
        for i in range(self.max_iterations):
            result = self.expand_neighbors(topic_entity, result)
        return result
    
    def expand_neighbors(self, topic_entity, entities):
        neighbors = set()
        for entity in entities:
            # get all the neighbor entities connected by relations
            relations = self.get_relations_between_entities(topic_entity, entity)
            for relation in relations:
                neighbors |= set(self.kg[relation][entity])
        # remove duplicates and entities already in the input
        neighbors -= set(entities) | {topic_entity}
        # compute the ComplEx scores for the new entities
        topic_embedding = self.kg_embedding_module.get_entity_embedding(topic_entity)
        scores = []
        for entity in neighbors:
            entity_embedding = self.kg_embedding_module.get_entity_embedding(entity)
            relation_embedding = self.kg_embedding_module.get_relation_embedding(self.get_relation_between_entities(topic_entity, entity))
            score = self.complex_score(topic_embedding, relation_embedding, entity_embedding)
            scores.append(score)
        # sort entities by score
        sorted_entities = [x for _, x in sorted(zip(scores, list(neighbors)), reverse=True)]
        # return the top k entities
        return sorted_entities[:self.k]
    
    def get_relations_between_entities(self, head_entity, tail_entity):
        # find all relations connecting head_entity and tail_entity
        relations = set()
        for r, e in self.kg.items():
            if head_entity in e and tail_entity in e:
                relations.add(r)
        return relations
    
    def get_relation_between_entities(self, head_entity, tail_entity):
        # find a relation connecting head_entity and tail_entity
        for r, e in self.kg.items():
            if head_entity in e and tail_entity in e:
                return r
        return None
    
    def complex_score(self, head_embedding, relation_embedding, tail_embedding):
        # compute the ComplEx score for a triple
        real_head, imag_head = np.split(head_embedding, 2)
        real_relation, imag_relation = np.split(relation_embedding, 2)
        real_tail, imag_tail = np.split(tail_embedding, 2)
        dot_real = np.dot(real_head * real_relation, real_tail) + np.dot(real_head * imag_relation, imag_tail)
        dot_imag = np.dot(imag_head * real_relation, imag_tail) - np.dot(imag_head * imag_relation, real_tail)
        return dot_real + dot_imag
