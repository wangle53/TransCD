import torch
import torch.nn as nn
from torch.nn import functional as F

##
def l1_loss(input, target):
    """ L1 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L1 distance between input and output
    """

    return torch.mean(torch.abs(input - target))

##
def l2_loss(input, target, size_average=True):
    """ L2 Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: L2 distance between input and output
    """
    if size_average:
        return torch.mean(torch.pow((input-target), 2))
    else:
        return torch.pow((input-target), 2)

def cos_loss(input, target, size_average=True):
    """ cosine Loss without reduce flag.

    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor

    Returns:
        [FloatTensor]: cosine distance between input and output
    """
    input = input.view(input.shape[0],-1)
    target = target.view(target.shape[0],-1)
    
    if size_average:
        return torch.mean(1-F.cosine_similarity(input, target))
    else:
        return 1-F.cosine_similarity(input, target)
    
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1
    
    def forward(self, predict, target):
#         target = target.unsqueeze(1)
#         print(predict.shape,target.shape)
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)
        
        pre = torch.sigmoid(predict).view(num, -1)
#         pre = predict.view(num, -1)
        tar = target.view(num, -1)
        
        intersection = (pre * tar).sum(-1).sum()  #鍒╃敤棰勬祴鍊间笌鏍囩鐩镐箻褰撲綔浜ら泦
        union = (pre + tar).sum(-1).sum()
        
        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)
        
        return score