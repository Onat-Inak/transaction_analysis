import wandb
from tqdm.notebook import tqdm
from tools.adjust_learning_rate import adjust_learning_rate

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

def train(model, loader, criterion, optimizer, scheduler, config):
    n_total_steps = len(loader)
    running_loss = 0.0
    average_loss = 0.0
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    seen_transactions = 0  # number of examples seen
    current_batch = 0
    for epoch in tqdm(range(config.num_epochs)):
        if config.lr_step_decay:
            optimizer = adjust_learning_rate(optimizer, epoch, config.learning_rate, config.lr_update_step)
        for i, (X_, y_) in enumerate(loader):
            seen_transactions += len(X_)
            current_batch += 1
            
            # Calculate loss in the corresponding batch
            loss = train_batch(X_, y_, model, optimizer, criterion, config.device)
            running_loss += loss.item()
            
            if (i+1) % config.log_step == 0:
                average_loss = running_loss/config.log_step
                print (f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {average_loss:.4f}')
                train_log(average_loss, current_batch, epoch, optimizer.param_groups[0]['lr'])
                running_loss, average_loss = 0.0, 0.0
        scheduler.step()


def train_batch(X_, y_, model, optimizer, criterion, device):
    X_, y_ = X_.to(device), y_.to(device)
    
    # Forward pass ➡
    outputs = model(X_)
    print(outputs.shape)
    loss = criterion(outputs, y_.view(-1, 1))
    
    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, current_batch, epoch, learning_rate):
    # Log to W&B
    wandb.log({"epoch": epoch+1, "loss": loss, "learning_rate": learning_rate}, step=current_batch)
    print("learning_rate: ", learning_rate)
    print(f"Loss after {str(current_batch)} batches: {loss:.3f}")