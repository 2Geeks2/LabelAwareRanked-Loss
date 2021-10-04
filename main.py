import numpy as np
import torch
import torch.nn as nn
from utils import BalancedBatchSampler, fit, extract_embeddings, plot_embeddings
from torch.optim import lr_scheduler
import torch.optim as optim
from networks import EmbeddingNet
from diff_losses import LabelAwareRankedLoss, AllTripletSelector, OnlineTripletLoss, NpairLossMod, ConstellationLoss
from torchvision.datasets import MNIST
from torchvision import transforms
import sys
import argparse

def main(loss, epochs):
    '''
    --loss: loss you want to choice
    --num_epochs: training epochs
    '''
    mean, std = 0.1307, 0.3081

    train_dataset = MNIST('../data/MNIST', train=True, download=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((mean,), (std,))
                          ]))
    test_dataset = MNIST('../data/MNIST', train=False, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((mean,), (std,))
                         ]))
    n_classes = 10

    cuda = torch.cuda.is_available()
    train_batch_sampler = BalancedBatchSampler(train_dataset.train_labels, n_classes=10, n_samples=2)
    test_batch_sampler = BalancedBatchSampler(test_dataset.test_labels, n_classes=10, n_samples=2)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # Set up the network and training parameters

    ''
    margin = 3.
    embedding_net = EmbeddingNet()
    model = embedding_net
    if cuda:
        model.cuda()

    if loss.lower() == 'triplet':
        loss_fn = OnlineTripletLoss(margin, AllTripletSelector())
    elif loss.lower() == 'npair':
        loss_fn = NpairLossMod()
    elif loss.lower() == 'constellation':
        loss_fn = ConstellationLoss(margin, AllTripletSelector())
    elif loss.lower() == 'lar':
        loss_fn = LabelAwareRankedLoss(margin, AllTripletSelector(), l2_reg=0.2)

    # loss_fn = NpairLoss(AllTripletSelector())
    # loss_fn = OnlineTripletLoss(margin, AllTripletSelector())
    # loss_fn = LabelAwareRankedLoss(margin, AllTripletSelector(), l2_reg=0.2)
    lr = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = epochs

    log_interval = 50
    fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

    train_embeddings_baseline, train_labels_baseline = extract_embeddings(train_loader, model)
    plot_embeddings(train_embeddings_baseline, train_labels_baseline)
    val_embeddings_baseline, val_labels_baseline = extract_embeddings(test_loader, model)
    plot_embeddings(val_embeddings_baseline, val_labels_baseline)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", help="name of the loss", default='lar')
    parser.add_argument("--num_epochs", help="number of epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.loss, args.num_epochs)
