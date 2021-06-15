import torch.nn as nn
import torch
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Adapted from:
    https://github.com/google-research/simclr/blob/master/objective.py
    https://github.com/facebookresearch/moco/blob/master/moco/builder.py
    with help from:
    https://github.com/AidenDurrant/SimCLR-Pytorch/blob/master/src/model/losses.py
    Args:
        init:
            temperature (float): temperature parameter of infoNCE loss
            batch_size (int, optional): batch size
        cosineSimilarity:
            embedding_1 (Tensor): representation output by enoder for transformations 1
                                  of size (batch_size, representation length)
            embedding_2 (Tensor): representation output by enoder for transformations 2
            of size (batch_size, representation length)
        call:
            embedding_1 (Tensor): representation output by enoder for transformations 1
                                  of size (batch_size, representation length)
            embedding_2 (Tensor): representation output by enoder for transformations 2
                                  of size (batch_size, representation length)
    Returns:
        cosineSimilarity:
            logits_11 (Tensor): temperature scaled cosine similarity matrix between tensors
                                embedding_1 & embedding_1
            logits_22 (Tensor):temperature scaled cosine similarity matrix between tensors
                               embedding_1 & embedding_1
            logits_12 (Tensor):temperature scaled cosine similarity matrix between tensors
                               embedding_1 & embedding_1
            logits_21 (Tensor):temperature scaled cosine similarity matrix between tensors
                               embedding_1 & embedding_1
        call:
            logits (Tensor): positive and negative similarities (pos, neg1, neg2, ...)
            labels (Tensor): labels (0, 0, ...)
            loss (Tensor): InfoNCE loss between two augmented versions of an image batch
                           (embedding_1, embedding_2)

    """

    def __init__(self, temperature, device, batch_size):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.t = temperature
        self.batch_size = batch_size

    def _cosine_similarity(self, embedding_1, embedding_2):
        norm_embedding_1 = F.normalize(embedding_1, dim=1)
        norm_embedding_2 = F.normalize(embedding_2, dim=1)
        torch.matmul(embedding_1, embedding_2.T)
        logits_11 = torch.mm(norm_embedding_1, norm_embedding_1.T) / self.t
        logits_22 = torch.mm(norm_embedding_2, norm_embedding_2.T) / self.t
        logits_12 = torch.mm(norm_embedding_1, norm_embedding_2.T) / self.t
        logits_21 = torch.mm(norm_embedding_2, norm_embedding_1.T) / self.t
        return logits_11, logits_22, logits_12, logits_21

    def __call__(self, embedding_1, embedding_2):
        # we assume embedding_1 and embedding_2 are already on the correct device on input
        mask = torch.eye(self.batch_size, dtype=torch.bool).to(self.device)
        logits_11, logits_22, logits_12, logits_21 = self._cosine_similarity(embedding_1, embedding_2)
        # discard the diagonals of logits_11 and logits_22 since these are the same embeddings
        # diagonals of logits_12 are the positive samples logits_21
        positives = torch.cat((logits_12[mask], logits_21[mask])).unsqueeze(1)
        # need to extract negatives from all logits
        neg_11 = logits_11[~mask].reshape(self.batch_size, -1)
        neg_22 = logits_22[~mask].reshape(self.batch_size, -1)
        neg_12 = logits_12[~mask].reshape(self.batch_size, -1)
        neg_21 = logits_21[~mask].reshape(self.batch_size, -1)
        neg_1 = torch.cat((neg_11, neg_12), dim=1)
        neg_2 = torch.cat((neg_22, neg_21), dim=1)
        negatives = torch.cat((neg_1, neg_2), dim=0)

        logits = torch.cat((positives, negatives), dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)
        loss = self.criterion(logits, labels).to(self.device)
        return logits, labels, loss
