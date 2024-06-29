import wandb

def adjust_learning_rate(optimizer, epoch, learning_rate, lr_update_step):
    lr = learning_rate * (0.1 ** (epoch+1 // lr_update_step))
    # wandb.log({"learning_rate": lr}, step=epoch+1)
    print(f'learning_rate: {lr:.6f}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer