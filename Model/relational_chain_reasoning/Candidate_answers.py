import networkx as nx

class RelationalChainReasoning:
    def __init__(self, kg, k=3):
        self.kg = kg
        self.k = k

    def _get_neighbors(self, entity):
        """Returns the neighbors of an entity in the KG"""
        neighbors = []
        for relation in self.kg[entity]:
            for neighbor in self.kg[entity][relation]:
                neighbors.append(neighbor)
        return neighbors

    def _expand_entities(self, entities):
        """Expands entities to include their neighbors up to k hops away"""
        expanded_entities = set()
        for entity in entities:
            neighbors = self._get_neighbors(entity)
            expanded_entities.update(neighbors)
        if self.k > 1:
            for i in range(2, self.k+1):
                for entity in expanded_entities:
                    neighbors = self._get_neighbors(entity)
                    expanded_entities.update(neighbors)
        return expanded_entities

    def _get_relation_score(self, entity, relation):
        """Calculates the relevance score of a relation to an entity"""
        score = 0
        for neighbor in self.kg[entity][relation]:
            score += 1 / len(self.kg[neighbor])
        return score

    def _get_top_relations(self, entity):
        """Returns the top relations for an entity"""
        relation_scores = []
        for relation in self.kg[entity]:
            score = self._get_relation_score(entity, relation)
            relation_scores.append((relation, score))
        top_relations = sorted(relation_scores, key=lambda x: x[1], reverse=True)[:self.k]
        return top_relations

    def _get_candidate_answers(self, entity, question):
        """Returns candidate answers based on relation matching"""
        top_relations = self._get_top_relations(entity)
        candidate_answers = set()
        for relation, score in top_relations:
            for answer in self.kg[entity][relation]:
                if answer == entity:
                    continue
                if answer in candidate_answers:
                    continue
                if relation in question:
                    candidate_answers.add(answer)
        return candidate_answers

    def get_best_answer(self, entity, question):
        """Returns the best answer to a question based on relation chain reasoning"""
        candidate_answers = self._get_candidate_answers(entity, question)
        expanded_entities = self._expand_entities(candidate_answers)
        best_answer = None
        best_score = -1
        for answer in expanded_entities:
            score = 0
            for relation in question:
                if relation in self.kg[entity]:
                    for neighbor in self.kg[entity][relation]:
                        if neighbor in self.kg[answer]:
                            score += 1 / len(self.kg[neighbor])
            if score > best_score:
                best_answer = answer
                best_score = score
        return best_answer
