import os
import shutil
import torch
from typing import List, Tuple

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name:str, fmt:str=':f') -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val:float, n:int=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches:int, meters:AverageMeter, prefix:str="") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch:int)->None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches:int)->str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output:torch.Tensor, target:torch.tensor, topk:Tuple[int]=(1,)) -> List[float]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
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
    
def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    try:
        torch.save(state, os.path.join(output_dir, filename))
        if is_best:
            shutil.copyfile(os.path.join(output_dir,filename), os.path.join(output_dir,'model_best.pth.tar'))
    except:
        print('Unable to save checkpoint to {} at this time...'.format(output_dir))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)