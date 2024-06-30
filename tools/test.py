import wandb
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def test(model, test_loader, device, project_name, save_model = False):
    model.eval()

    with torch.no_grad():
        f1_score_test = 0.0
        total_y_ = torch.tensor([]).to(device)
        total_predictions = torch.tensor([]).to(device)
        correct, total = 0, 0
        for batch, (X_, y_) in enumerate(test_loader):
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            predictions = torch.round(outputs)
            total_y_ = torch.cat((total_y_, y_.view(-1,1)))
            total_predictions = torch.cat((total_predictions, predictions))
            total += y_.size(0)
            correct += (predictions == y_.view(-1,1)).sum().item()
            
        cm = confusion_matrix(total_y_.to('cpu'), total_predictions.to('cpu'))
        ConfusionMatrixDisplay(cm).plot()
        f1_score_test = f1_score(total_y_.to('cpu'), total_predictions.to('cpu'))
        
        # Print and log the results
        print(f"Accuracy of the model on {total} " +
              f"transactions in test data: {correct / total:%}")
        print('f1_score_test: ', f1_score_test)
        print(classification_report(total_y_.to('cpu'), total_predictions.to('cpu')))
        
        wandb.log({"f1_score_test": f1_score_test})
        wandb.log({"test_accuracy": correct / total})

    if save_model:
        # Save the model in the exchangeable ONNX format
        torch.save(model.state_dict(), 'experiments/'+project_name+'/model_state_dict.pth')
        # torch.save(model, 'experiments/RNN/model.pt')
        # wandb.unwatch()
        wandb.save('experiments/'+project_name+'/model_state_dict.pth')
        # wandb.save('experiments/RNN/model.pt')