import torch
import numpy as np
import torch.nn as nn

def retrieval_ref(model, batch_query, extra_memory_loader, evaluation, device):
    """
    Given a batch of query and a extra memory, to search a best matched img ref
    Params:
        model(torch.nn.Module): for extract feature
        batch_query(torch.Tensor): a batch of query
        extra_memory_loader(torch.data.DataLoader): A dataset loader
        evaluation(torch.nn.Module or other function for evaluation distance):
            typically cosine distance or l2 distance
        device(torch.device): move the model and data to gpu or cpu
    Return: a btach of img, the best matched img for queries
    """
    batch_query = batch_query.to(device)
    model = model.to(device)
    batch_qf = model(batch_query)
    best_min_val, best_min_img = torch.zeors(batch_qf.size(0)), torch.zeros_like(batch_query)
    for i, data in enumerate(extra_memory_loader):
        imgs = data[0]
        imgs = imgs.to(device)
        ref_f = model(imgs)
        # a n*n matrix each column represent the distance
        # between ith query and each image in memeory
        dist = evalution(batch_qf, ref_f)

        min_val, min_ind = torch.min(dist, 1)
        # Update the best min val
        min_cmp = min_val < best_min_val
        best_min_val[min_cmp] = min_val[min_cmp]
        best_min_img[min_cmp] = imgs[min_ind[min_cmp]]

    return best_min_img
