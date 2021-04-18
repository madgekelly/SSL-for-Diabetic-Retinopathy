import torch
from tqdm.notebook import tqdm
import csv


class PreTrainSimCLR:

    def __init__(self, criterion, optimiser, device):
        self.criterion = criterion
        self.optimiser = optimiser
        self.device = device

    def accuracy(self, output, target, topk=(1,)):
        """
        taken from: https://github.com/facebookresearch/moco/blob/master/main_moco.py
        Computes the accuracy over the k top predictions for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def train_multiple_epochs(self, encoder, data_loader, num_epochs, folder, start=0):
        for i in range(start, num_epochs):
            print('epoch {} of {}'.format(i, num_epochs))
            loss, acc_1, acc_2, encoder = self.train_epoch(data_loader)

            # save to history every epoch
            with open('{}/history.csv'.format(folder), 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([loss, acc_1, acc_2])

            # save model every 5 epochs
            if (i % 5 == 0) | (i == num_epochs - 1):
                torch.save(encoder, '{}/epoch_{}'.format(folder, i))

    def train_epoch(self, encoder, data_loader):
        no_batches = len(data_loader)
        progress_bar = tqdm(total=no_batches, desc='Batch', position=0)
        loss_sum = 0
        acc_sum_1 = 0
        acc_sum_2 = 0
        for i, (images, labels) in enumerate(data_loader):
            embedding_1 = encoder(images[0].to(self.device))
            embedding_2 = encoder(images[1].to(self.device))
            logits, labels, loss = self.criterion(embedding_1, embedding_2)

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            loss_sum += loss.item()
            top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
            acc_sum_1 += top1
            acc_sum_2 += top5
            progress_bar.set_postfix({'loss': loss.item()})
            progress_bar.update(1)

        loss = loss_sum / no_batches
        acc_1 = acc_sum_1 / no_batches
        acc_2 = acc_sum_2 / no_batches

        return float(loss), float(acc_1), float(acc_2), encoder.get_model()