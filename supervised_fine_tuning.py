import torch
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# class LogisticRegression(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegression, self).__init__()
#         self.linear = torch.nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         outputs = self.linear(x)
#         return outputs
#
# class kmeans:
#
#     def __init__(self, output_dim):
#         self.kmeans = KMeans(n_clusters=output_dim, random_state=0)
#
#     def fit_predict(self):
#         self.kmeans.fit()
#         preds = kmeans.predict()
#
#     def get_labels(self, predictions, labels):
#         clusters = np.array([predictions, labels])
#         cluster_df = pd.DataFrame(clusters.T, columns=['cluster', 'label'])
#         grouped_clusters = cluster_df.groupby(['cluster', 'label']).agg({'label': ['count']}).reset_index()
#         grouped_clusters.columns = ['cluster', 'label', 'count']
#         cluster_values = grouped_clusters.sort_values('count', ascending=False).drop_duplicates('cluster', keep='first')[['cluster', 'label']]
#         predictions = pd.DataFrame(clusters.T, columns=['predictions', 'true_labels'])
#         return predictions.merge(cluster_values, left_on='predictions', right_on='cluster')[['true_labels', 'label']]
#
#
#
# def train_linear_classifier(embedding_loader, classifier, criterion, optimiser, num_epochs, history=None, weights=None):
#     no_batches = len(embedding_loader)
#     accuracies = []
#     losses = []
#     for epoch in range(num_epochs):
#         print("Epoch {} of {}".format(epoch, num_epochs))
#         cum_loss = 0
#         cum_acc = 0
#         total = 0
#         for i, (embedding, labels) in enumerate(embedding_loader):
#             optimiser.zero_grad()
#             outputs = classifier(embedding)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimiser.step()
#
#             _, predicted = torch.max(outputs.data, 1)
#             correct = int((predicted == labels).sum())
#             cum_acc += correct
#             total += len(labels)
#             cum_loss += loss.item()
#
#         loss = cum_loss / no_batches
#         losses.append(loss)
#         acc = cum_acc / total
#         accuracies.append(acc)
#
#         # track epoch performance
#         if history is not None:
#
#             with open(history, a+, newline='') as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow([loss, acc])
#     if weights is not None:
#         torch.save(classifier, weights)
#
#     return classifier