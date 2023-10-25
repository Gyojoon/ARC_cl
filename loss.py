import torch
import torch.nn as nn

def nt_xent_loss(output, temperature):
    batch_size = output.shape[0]  
    logits = torch.mm(output, output.t().contiguous()) / temperature
    labels = torch.arange(batch_size).to(output.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss

def our_loss(output, labels, temperature):
    batch_size = output.shape[0]  
    logits = torch.mm(output, output.t().contiguous()) / temperature
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def soft_loss(output, labels, temperature):
    # 구현 중
    batch_size = output.shape[0]  
    logits = torch.mm(output, output.t().contiguous()) / temperature
    # topk_indices = torch.topk(logits, target_n, dim=1).indices.to(output.device)
    # label_mask = torch.zeros(batch_size, batch_size).to(output.device)
    # for i, topk in enumerate(topk_indices):
    #     label_mask[i][topk] = 1 / target_n
    # label = torch.arange(batch_size).to(output.device)
    loss = nn.functional.cross_entropy(logits, labels)
    return loss