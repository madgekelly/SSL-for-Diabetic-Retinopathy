import torch
from tqdm.notebook import tqdm
import csv
from losses.infoNCE_loss import InfoNCELoss
from models.simCLR_encoders import SimCLREncoder
from models.optimisers import get_optimiser
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
from torch.utils.data import DataLoader
from data.data_loader import DataSetFromFolder
from data.data_transforms import SimCLRDataSetTransform
from data.set_seed import set_seeds
import argparse
import json
import torch.distributed as dist

parser = argparse.ArgumentParser(description='simCLR pre-training')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('image_path',
                    help='path to dataset')
parser.add_argument('label_path',
                    help='path to dataset')
parser.add_argument('save_folder',  type=str,
                    help='location to save results')
parser.add_argument('--encoder_type', default='resnet18', type=str,
                    help='type of encoder to use')
parser.add_argument('--num_epochs', default=1, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='the epoch to start training on')
parser.add_argument('-lr', '--learning_rate', default=0.3, type=float,
                    help='initial learning rate')
parser.add_argument('-t', '--temp', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('-momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-wd', '--weight_decay', default=0., type=float,
                    help='weight decay (default: 0.)')
parser.add_argument('-p', '--save-freq', default=10, type=int,
                    help='frequency to save model weights (number of epochs)')
parser.add_argument('--optimiser', default="SGD", type=str,
                    choices=["SGD", "LARS"],
                    help='optimiser to be used in training')
parser.add_argument('--warm_starts', default=10, type=int,
                    help='number of epochs for warms starts and cosine decacy')
parser.add_argument('--aug', default=None,
                    help='augmentation strategy number')

# data parallel arguments
parser.add_argument('--DDP',  type=bool, default=False,
                    help='whether to use distributed data parallel')
parser.add_argument('--local_rank', type=int, default=0)


class PreTrainSimCLR:

    def __init__(self, criterion, optimiser, encoder, scheduler, device):
        """
        :param criterion: the loss function
        :param optimiser: the model optimiser
        :param encoder: the model to train from class simCLREncoder, inheriting from torch.nn.Module
        :param device: device ID for the model to be trained on
        """
        self.criterion = criterion
        self.device = device
        self.encoder = encoder
        self.optimiser = optimiser
        self.scheduler = scheduler

    def accuracy(self, output, target, topk=(1,)):
        """
        taken from: https://github.com/facebookresearch/moco/blob/master/main_moco.py
        Computes the accuracy over the k top predictions for the specified values of k
        :param output: logits (model predictions)
        :param target: labels (true labels)
        :param topk: a tuple of which top-k accuracies to return
        :return:
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

    def train_multiple_epochs(self, data_loader, num_epochs, save_freq, folder, args, start=0):
        """
        :param data_loader: a pytorch data loader of type image, label
        :param num_epochs: an integer that determines the number of epochs to train the model
        :param save_freq: an integer that determines the frequency of saving the model (every n epochs)
        :param folder: a filepath determining the location of the folder to save the models and train history
        :param start: at which epoch the training is starting
        :return:
        """

        for i in range(start, num_epochs):
            print('epoch {} of {}'.format(i, num_epochs))
            loss, acc_1, acc_2 = self.train_epoch(data_loader)
            if self.scheduler is not None:
                if i + 1 <= args.warm_starts:
                    self.scheduler.step()
            print('loss: {} top-1 acc: {} top-5 acc: {}'.format(loss, acc_1, acc_2))

            # save to history every epoch
            with open('{}/history.csv'.format(folder), 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([loss, acc_1, acc_2])

            # save model every save_freq epochs
            if (i % save_freq == 0) | (i == num_epochs - 1):
                if args.DDP:
                    if dist.get_rank() == 0:
                        torch.save(self.encoder.get_model().state_dict(), '{}/epoch_{}'.format(folder, i))
                        torch.save(self.optimiser.state_dict(), '{}/optimiser_epoch_{}'.format(folder, i))
                else:
                    torch.save(self.encoder.get_model().state_dict(), '{}/epoch_{}'.format(folder, i))
                    torch.save(self.optimiser.state_dict(), '{}/optimiser_epoch_{}'.format(folder, i))

    def train_epoch(self, data_loader, progress_bar=False):
        """
        :param data_loader: pytorch DataLoader with format (image, label)
        :param progress_bar: Boolean determining whether a progress bar is returned to track model progress
        :return: loss, top-1 accuracy, top-5 accuracy
        """
        no_batches = 0
        if progress_bar:
            progress_bar = tqdm(total=no_batches, desc='Batch', position=0)
        loss_sum = 0
        acc_sum_1 = 0
        acc_sum_2 = 0
        for i, (images, labels) in enumerate(data_loader):
            images = torch.cat(images, dim=0)
            images = images.to(self.device)
            embeddings = self.encoder(images)
            embedding_1, embedding_2 = torch.tensor_split(embeddings, 2, dim=0)
            logits, labels, loss = self.criterion(embedding_1, embedding_2)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            loss_sum += loss.item()
            top1, top5 = self.accuracy(logits, labels, topk=(1, 5))
            acc_sum_1 += top1
            acc_sum_2 += top5
            if progress_bar:
                progress_bar.set_postfix({'loss': loss.item()})
                progress_bar.update(1)
            no_batches += 1
        loss = loss_sum / no_batches
        acc_1 = acc_sum_1 / no_batches
        acc_2 = acc_sum_2 / no_batches

        return float(loss), float(acc_1), float(acc_2)


def main():
    args = parser.parse_args()

    with open(args.save_folder + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    set_seeds(0)
    if args.DDP:
        init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        torch.autograd.set_detect_anomaly(True)
        device = torch.device("cuda", args.local_rank)
    else:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    transform = SimCLRDataSetTransform(s=1, size=224, strategy=args.aug)
    data = DataSetFromFolder(args.image_path, args.label_path, transform, index=False, mode='pretrain')
    # note DistributedSampler shuffles by default

    if args.DDP:
        data_sampler = DistributedSampler(data)
        data_loader = DataLoader(data, batch_size=args.batch_size, num_workers=0, pin_memory=True, sampler=data_sampler, drop_last=True)
        encoder = SimCLREncoder(args.encoder_type, 128, device, local_rank=args.local_rank)
    else:
        data_loader = DataLoader(data, batch_size=args.batch_size, drop_last=True)
        encoder = SimCLREncoder(args.encoder_type, 128, device, DDP=False)

    optimiser = get_optimiser(args, encoder)
    scheduler = None
    # only use scheduler for LARS
    if args.optimiser == "LARS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, args.num_epochs - args.warm_starts)
    criterion = InfoNCELoss(args.temp, device, args.batch_size)
    pretrain_init = PreTrainSimCLR(criterion, optimiser, encoder, scheduler, device)
    pretrain_init.train_multiple_epochs(data_loader, args.num_epochs, args.save_freq, args.save_folder, args)


if __name__ == '__main__':
    main()
