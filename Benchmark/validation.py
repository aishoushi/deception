import torch
from torch.autograd import Variable
import time
import sys
import torch.nn.functional as F
from utils import AverageMeter, calculate_accuracy, calculate_auc


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))
    outputs_epoch = torch.FloatTensor([1.0])
    targets_epoch = torch.LongTensor([1.0])   

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    aucs = AverageMeter()
    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
            inputs = inputs.cuda(async=True)
        #print(inputs.shape)
        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)
        auc = calculate_auc(outputs[:,0].cpu(), targets.cpu())
        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))
        aucs.update(auc, inputs.size(0))
        batch_time.update(time.time() - end_time)
        end_time = time.time()

       # if not opt.no_softmax_in_test:
       #     outputs = F.softmax(outputs)
       # outputs_epoch = torch.cat((outputs_epoch, outputs[:,0].cpu()), dim=0)
 #       print(outputs_epoch.shape, targets_epoch.shape)
       # targets_epoch = torch.cat((targets_epoch, targets.cpu()), dim=0)
  #    #  print(outputs_epoch.shape, targets_epoch.shape)
       # print('batch:', i, 'acc:', acc)
        del outputs, targets
   # outputs_epoch = outputs_epoch[1:-1]
   # targets_epoch = targets_epoch[1:-1]
    
   # auc = calculate_auc(outputs_epoch, targets_epoch)
   # print('auc', auc, accuracies.avg) 

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
              'Auc {auc.val:.3f} ({auc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies,
                  auc=aucs))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'auc': aucs.avg})

    return losses.avg
