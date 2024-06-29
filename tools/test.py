import wandb
import torch

def test(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for X_, y_ in test_loader:
            X_, y_ = X_.to(device), y_.to(device)
            outputs = model(X_)
            predicted = torch.round(outputs)
            print("predicted: ", torch.transpose(predicted[:10], 0, 1))
            print("y_: ", torch.transpose(y_.view(-1,1)[:10], 0, 1))
            total += y_.size(0)
            correct += (predicted == y_.view(-1,1)).sum().item()

        print(f"Accuracy of the model on the {total} " +
              f"transactions in test data: {correct / total:%}")
        
        wandb.log({"test_accuracy": correct / total})

    # Save the model in the exchangeable ONNX format
    torch.save(model.state_dict(), 'experiments/RNN/model_state_dict.pth')
    # torch.save(model, 'experiments/RNN/model.pt')
    # wandb.unwatch()
    wandb.save('experiments/RNN/model_state_dict.pth')
    # wandb.save('experiments/RNN/model.pt')