import wandb
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def validate(model, val_loader, device):
    val_accuracy = 0.0
    f1_score_val = 0.0
    with torch.no_grad():
        correct, total = 0, 0
        for X_, y_ in val_loader:
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            predictions = torch.round(outputs)
            # print("predictions: ", torch.transpose(predictions[:10], 0, 1))
            # print("y_: ", torch.transpose(y_.view(-1,1)[:10], 0, 1))
            total += y_.size(0)
            correct += torch.sum(predictions == y_.view(-1,1))
            f1_score_val = f1_score(y_.view(-1,1).to('cpu'), predictions.to('cpu'))
        val_accuracy = correct / total 
        print('\n Validation:')
        print(classification_report(y_.view(-1,1).to('cpu'), predictions.to('cpu')))
    return val_accuracy, f1_score_val
