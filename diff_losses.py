import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        #print ("labels")
        #print (labels)
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class ConstellationLoss(nn.Module):
    """
    Constellation Loss
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets, in our function, we select all triplets that mean hyperparameter K = numb_class - 1
    """
    def __init__(self, margin, triplet_selector):
        super(ConstellationLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        num_class = len(np.unique(target))
        n = triplets.shape[0]
        anchors = embeddings[triplets[:, 0]]
        positives = embeddings[triplets[:, 1]]
        negatives = embeddings[triplets[:, 2]]
        anchors = torch.unsqueeze(anchors, dim=1)
        positives = torch.unsqueeze(positives, dim=1)
        negatives = torch.unsqueeze(negatives, dim=1)
        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))
        x = torch.exp(x.squeeze())
        x = torch.sum(torch.reshape(x, (num_class, -1)), axis=1)
        loss = torch.log(1 + x)

        return loss.mean()


class NpairLossMod(nn.Module):
    """
    Multi-class n-pair loss
    Using smart batch: [f0, f0', f1, f1', ..., fc, fc'], where c is the maximum label
    shape for embedding should be [2c, d], where d is the dim of embedding vector
    """

    def __init__(self, l2_reg=0.02):
        super(NpairLossMod, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):

        anchor = embeddings[::2, :]
        positive = embeddings[1::2, :]
        target = target[::2]
        batch_size = anchor.size(0)

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        logit = torch.exp(logit - torch.diag(logit))

        loss_ce = torch.log(torch.sum(logit, dim=1)).mean()
        l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss


class LabelAwareRankedLoss(nn.Module):
    '''
    Label-Aware Ranked Loss
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    Using smart batch: [f0, f0', f1, f1', ..., fc, fc'], where c is the maximum label
    Multipliers are calculated as 'diff'
    '''
    def __init__(self, margin, triplet_selector, l2_reg=0.02):
        super(LabelAwareRankedLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.l2_reg = l2_reg

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        n = triplets.shape[0]
        num_class = len(np.unique(target))
        # print(num_class)
        # input('here')
        if embeddings.is_cuda:
            triplets = triplets.cuda()
            num_class = num_class.cuda()
        tar_anchors = target[triplets[:, 0]]
        tar_positives = target[triplets[:, 1]]
        tar_negatives = target[triplets[:, 2]]

        diff = torch.min(torch.abs(tar_anchors - tar_negatives),
                         torch.abs(num_class - torch.abs(tar_anchors - tar_negatives)))
        diff = torch.log(diff)

        if embeddings.is_cuda:
            diff = diff.cuda()

        anchors = embeddings[triplets[:, 0]]
        positives = embeddings[triplets[:, 1]]
        negatives = embeddings[triplets[:, 2]]
        anchors = torch.unsqueeze(anchors, dim=1)
        positives = torch.unsqueeze(positives, dim=1)
        negatives = torch.unsqueeze(negatives, dim=1)

        diff = diff.view(diff.shape[0], 1)
        diff = torch.unsqueeze(diff.repeat((1, 2)), dim=1)
        x = torch.matmul(anchors, (diff * negatives - positives).transpose(1, 2))
        x = torch.exp(x.squeeze())
        x = torch.sum(torch.reshape(x, (num_class, -1)), axis=1)

        loss = torch.log(1 + x)  # / 1000
        l2_loss = torch.sum(anchors ** 2) / n + torch.sum(positives ** 2) / n + torch.sum(negatives ** 2) / n
        return loss.mean() + l2_loss * 0.25 * self.l2_reg