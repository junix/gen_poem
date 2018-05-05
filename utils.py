def change_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr
    return optimizer
