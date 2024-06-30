import wandb
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

def validate(model, val_loader, device):
    with torch.no_grad():
        val_accuracy = 0.0
        f1_score_val = 0.0
        total_y_ = torch.tensor([]).to(device)
        total_predictions = torch.tensor([]).to(device)
        correct, total = 0, 0
        for batch, (X_, y_) in enumerate(val_loader):
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            predictions = torch.round(outputs)
            total_y_ = torch.cat((total_y_, y_.view(-1,1)))
            total_predictions = torch.cat((total_predictions, predictions))
            # print("predictions: ", torch.transpose(predictions[:10], 0, 1))
            # print("y_: ", torch.transpose(y_.view(-1,1)[:10], 0, 1))
            total += y_.size(0)
            correct += torch.sum(predictions == y_.view(-1,1))
        f1_score_val = f1_score(total_y_.to('cpu'), total_predictions.to('cpu'))
        val_accuracy = correct / total 
        print('\n Validation:')
        print(classification_report(total_y_.to('cpu'), total_predictions.to('cpu')))
    return val_accuracy, f1_score_val
