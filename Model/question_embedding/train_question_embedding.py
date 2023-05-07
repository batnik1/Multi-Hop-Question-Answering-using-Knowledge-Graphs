from question_embedding import QuestionEmbedding
import torch.optim as optim
import torch.nn as nn


def train(question_embedder, train_data, num_entities, learning_rate, num_epochs):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(question_embedder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for question, topic_entity, candidate_entities, labels in train_data:
            optimizer.zero_grad()

            scores = question_embedder(question, topic_entity, candidate_entities)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_data)
        print('Epoch {}: Loss = {:.4f}'.format(epoch+1, epoch_loss))


if __name__ == '__main__':
    train_data = [
        ('What is the capital of France?', 1, [2, 3, 4], [0, 0, 1]),
        ('What is the largest country in the world?', 5, [6, 7, 8], [1, 0, 0]),
        ('Who wrote the book "1984"?', 9, [10, 11, 12], [1, 0, 0])
    ]

    num_entities = 20
    embedding_dim = 50
    learning_rate = 0.001
    num_epochs = 10

    question_embedder = QuestionEmbedding(num_entities, embedding_dim)
    train(question_embedder, train_data, num_entities, learning_rate, num_epochs)
