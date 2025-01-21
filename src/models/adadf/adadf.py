import numpy as np
import torch

def generate_adaptive_LD(outputs, targets, num_classes, threshold, sharpen, T):
    device = outputs.device
    LD = torch.zeros(num_classes, num_classes).to(device)

    outputs_l = outputs[1:, :]
    targets_l = targets[1:]

    probs = torch.softmax(outputs_l, dim=1)

    for i in range(num_classes):
        idxs = np.where(targets_l.cpu().numpy()==i)[0]
        if torch.mean(probs[idxs], dim=0)[i] >= threshold:
            LD[i] = torch.mean(probs[idxs], dim=0)
        else:
            LD[i] = torch.zeros(num_classes).fill_(0.4/(num_classes-1)).scatter_(0, torch.tensor(i), threshold)
        # LD[i] = torch.zeros(num_classes).fill_(0.4/(num_classes-1)).scatter_(0, torch.tensor(i), 0.6)

    if sharpen == True:
        LD = torch.pow(LD, 1/T) / torch.sum(torch.pow(LD, 1/T), dim=1)

    return LD