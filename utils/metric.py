import torch

def all_metric(outputs, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for id, label in enumerate(labels):
        output = outputs[id]
        if label == 1:
            if output == 1:
                tp += 1
            else:
                fn += 1
        else:
            if output == 0:
                tn += 1
            else:
                fp += 1
    accu = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accu, precision, recall, f1

def accuracy(outputs: torch.Tensor, labels: torch.Tensor):
    num_all = 0
    num_hits = 0
    limit = 0.5
    for i in range(outputs.size(0)):
        if outputs[i] > limit:
            pred = 1
        else:
            pred = 0
        if labels[i] == pred:
            num_hits += 1
        num_all += 1
    return num_hits, num_all, num_hits / num_all