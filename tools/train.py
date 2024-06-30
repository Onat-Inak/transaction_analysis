import wandb
import torch
from tqdm.notebook import tqdm
from tools.adjust_learning_rate import adjust_learning_rate
from tools.validate import validate
from sklearn.metrics import f1_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, config):
    n_total_steps = len(train_loader)
    running_train_loss = 0.0
    average_train_loss = 0.0
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    seen_transactions = 0  # number of examples seen
    current_batch = 0
    for epoch in tqdm(range(config.num_epochs)):
        model.train()
        if config.lr_step_decay:
            optimizer = adjust_learning_rate(optimizer, epoch, config.learning_rate, config.lr_update_step)
        for i, (X_, y_) in enumerate(train_loader):
            seen_transactions += len(X_)
            current_batch += 1
            
            # Calculate training loss in the corresponding batch
            train_loss = train_batch(X_, y_, model, optimizer, criterion, config.device)
            running_train_loss += train_loss.item()
            
            # Log first training loss
            if epoch == 0 and i == 0:
                wandb.log({"train_loss": train_loss}, step=current_batch)
                model.eval()
                val_accuracy, f1_score_val = validate(model, val_loader, config.device)
                model.train()
                wandb.log({"val_accuracy": val_accuracy, "epoch": epoch+1, "f1_score_val": f1_score_val})
            
            if (i+1) % config.log_step == 0:
                average_train_loss = running_train_loss/config.log_step
                print (f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{n_total_steps}], Train Loss: {average_train_loss:.4f}')
                # Log training metrics
                train_log(average_train_loss, current_batch, epoch, optimizer.param_groups[0]['lr'])
                running_train_loss, average_train_loss = 0.0, 0.0
        # Calculate validation accuracy
        model.eval()
        val_accuracy, f1_score_val = validate(model, val_loader, config.device)
        model.train()
        wandb.log({"val_accuracy": val_accuracy, "epoch": epoch+1, "f1_score_val": f1_score_val})
        print(f"Validation Accuracy: {val_accuracy:%}")
        print('')
        # Decrease the learning rate regarding config.gamma
        scheduler.step()


def train_batch(X_, y_, model, optimizer, criterion, device):
    X_, y_ = X_.to(device), y_.to(device)
    
    # Forward pass ➡
    outputs = model(X_)
    train_loss = criterion(outputs.to(device), y_.view(-1, 1))
    
    # Backward pass ⬅
    optimizer.zero_grad()
    train_loss.backward()

    # Step with optimizer
    optimizer.step()

    return train_loss

def train_log(train_loss, current_batch, epoch, learning_rate):
    # Log to W&B
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "learning_rate": learning_rate}, step=current_batch)
    print("learning_rate: ", learning_rate)
    print(f"Train Loss after {str(current_batch)} batches: {train_loss:.3f}")
    