import torch
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy


def dice_loss(predictions, labels):
    """ based on loss function from V-Net paper """
    softmaxed = softmax(predictions, 1)
    predictions = softmaxed[:, 1, :]  # just the root probability.
    labels = labels.float()
    preds = predictions.contiguous().view(-1)
    labels = labels.view(-1)
    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))


def combined_loss(predictions, labels):
    """ mix of dice and cross entropy """
    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    if torch.sum(labels) > 0:
        return (dice_loss(predictions, labels) +
                (0.3 * cross_entropy(predictions, labels)))
    # When no roots use only cross entropy
    # as dice is undefined.
    return 0.3 * cross_entropy(predictions, labels)


## Ideas for next versions of these loss functions. 

def dice_loss2(preds, labels):
    """ based on loss function from V-Net paper """
    assert torch.max(labels) <= 1
    assert torch.min(labels) >= 0
    assert torch.max(preds) <= 1
    assert torch.min(preds) >= 0

    intersection = torch.sum(torch.mul(preds, labels))
    union = torch.sum(preds) + torch.sum(labels)
    return 1 - ((2 * intersection) / (union))

def combined_loss2(preds, labels, mask=None):
    """ mix of dice and cross entropy """
    if mask is not None:
        preds  = torch.mul(preds, mask) # weighted by defined region of annotation

    # if they are bigger than 1 you get a strange gpu error
    # without a stack track so you will have no idea why.
    assert torch.max(labels) <= 1
    cx = (0.3 * binary_cross_entropy(preds, labels))
    if torch.sum(labels) > 0:
        dl = dice_loss(preds, labels)
        return dl + cx
    return cx