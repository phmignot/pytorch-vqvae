import numpy as np
import torch

def getIndexPix(pixel):
    return (pixel[0].item(), pixel[1].item(), pixel[2].item())

def buildPixDictionary(input):
    pixDictionary = {}
    print("input size",input.size())
    batch, channel, height, width = input.size()
    for img in range(batch):
        for i in range(height):
            for j in range(width):
                key = getIndexPix(input[img,:,i,j].detach())
                if key in pixDictionary:
                    pixDictionary[key] += 1
                else:
                    pixDictionary[key] = 1
    return pixDictionary


def color_loss(input, target, reduction='mean'):
    pixDictionary = buildPixDictionary(target)
    print("Len pixDict", len(pixDictionary))
    print(pixDictionary)
    print(input.size())
    batch, channel, height, width = input.size()
    ret = torch.empty(batch, channel, height, width)
    nbPixelDif = len(pixDictionary)
    nbPixel = height * width * batch
    if target.requires_grad:
        for img in range(batch):
            for i in range(height):
                for j in range(width):
                    mse = (input[img,:,i,j] - target[img,:,i,j]) ** 2
                    nbApparition = pixDictionary[getIndexPix(input[img,:,i,j].detach())]
                    color_coef = nbPixel / (nbPixelDif * nbApparition)
                    ret[img,:,i,j] = mse * color_coef
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        for i in range(height):
            for j in range(width):
                mse = (expanded_input[i][j] - expanded_target[i][j]) ** 2
                nbApparition = pixDictionary[input[i][j]]
                color_coef = nbPixel / (nbPixelDif * nbApparition)
                ret[i][j] = mse * color_coef
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)


def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
        print("OUPS ")
    if target.requires_grad:
        ret = (input - target) ** 2
        print("OK ")
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        print("======WOW=================================")
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = (expanded_input - expanded_target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret
