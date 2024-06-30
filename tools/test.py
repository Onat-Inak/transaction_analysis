import wandb
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def test(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for X_, y_ in test_loader:
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            # print("outputs.shape: ", outputs.shape)
            predictions = torch.round(outputs)
            # print("predictions: ", torch.transpose(predictions[:20], 0, 1))
            # print("y_: ", torch.transpose(y_.view(-1,1)[:20], 0, 1))
            total += y_.size(0)
            correct += (predictions == y_.view(-1,1)).sum().item()
            cm = confusion_matrix(y_.view(-1,1).to('cpu'), predictions.to('cpu'))
            ConfusionMatrixDisplay(cm).plot()
            # f1 = f1_score(y_.view(-1,1).to('cpu'), predictions.to('cpu'))
        print(f"Accuracy of the model on {total} " +
              f"transactions in test data: {correct / total:%}")
        # print('F1-Score: ', f1)
        print(classification_report(y_.view(-1,1).to('cpu'), predictions.to('cpu')))
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), 'experiments/RNN/model_state_dict.pth')
    # torch.save(model, 'experiments/RNN/model.pt')
    # wandb.unwatch()
    wandb.save('experiments/RNN/model_state_dict.pth')
    # wandb.save('experiments/RNN/model.pt')