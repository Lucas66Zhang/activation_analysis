import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.LeNet5 import LeNet5

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

def main():
    train_dataset = datasets.MNIST(root='../data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # optimizers = [optim.Adam, optim.SGD]
    # learning_rates = [5e-3, 1e-2, 5e-2, 1e-1]
    # regularizations = [0, 5e-2, 1e-1, 2e-1]  # L2
    # epochs_list = [1,2,3,4]
    # seeds = [0, 1, 2, 42]
    # initializations = ['xavier', 'normal', 'kaiming', 'default']
    # batch_sizes = [32, 64, 128, 256]
    # device = torch.device('mps')
    # n_iter = 4*4*4*4*4*4*2
    optimizers = [optim.SGD, optim.Adam]
    learning_rates = [5e-3, 1e-2, 5e-2, 1e-1]
    regularizations = [0, 5e-2, 1e-1, 2e-1]  # L2
    epochs_list = [1,2,3,4]
    seeds = [0, 1, 2, 42]
    initializations = ['xavier', 'normal', 'kaiming', 'default']
    batch_sizes = [32, 64, 128, 256]
    device = torch.device('mps')
    n_iter = 4**6 * 2

    pbar = tqdm(total=n_iter, position=0, leave=True)
    for optimizer in optimizers:
        for lr in learning_rates:
            for reg in regularizations:
                for epochs in epochs_list:
                    for seed in seeds:
                        for init in initializations:
                            for batch_size in batch_sizes:
                                torch.manual_seed(seed)

                                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

                                model = LeNet5().to(device)

                                # Initialization
                                if init == 'xavier':
                                    torch.nn.init.xavier_uniform_(model.fc1.weight)
                                    torch.nn.init.xavier_uniform_(model.fc2.weight)
                                    torch.nn.init.xavier_uniform_(model.fc3.weight)
                                    torch.nn.init.xavier_uniform_(model.conv1.weight)
                                    torch.nn.init.xavier_uniform_(model.conv2.weight)
                                elif init == 'normal':
                                    torch.nn.init.normal_(model.fc1.weight, mean=0, std=0.01)
                                    torch.nn.init.normal_(model.fc2.weight, mean=0, std=0.01)
                                    torch.nn.init.normal_(model.fc3.weight, mean=0, std=0.01)
                                    torch.nn.init.normal_(model.conv1.weight, mean=0, std=0.01)
                                    torch.nn.init.normal_(model.conv2.weight, mean=0, std=0.01)
                                elif init == 'kaiming':
                                    torch.nn.init.kaiming_uniform_(model.fc1.weight)
                                    torch.nn.init.kaiming_uniform_(model.fc2.weight)
                                    torch.nn.init.kaiming_uniform_(model.fc3.weight)
                                    torch.nn.init.kaiming_uniform_(model.conv1.weight)
                                    torch.nn.init.kaiming_uniform_(model.conv2.weight)
                                else:
                                    pass

                                opt = optimizer(model.parameters(), lr=lr, weight_decay=reg)

                                criterion = nn.CrossEntropyLoss()

                                # Train
                                for epoch in range(epochs):
                                    model.train()
                                    for batch_idx, (data, target) in enumerate(train_loader):
                                        data, target = data.to(device), target.to(device)
                                        opt.zero_grad()
                                        outputs = model(data)
                                        loss = criterion(outputs, target)
                                        loss.backward()
                                        opt.step()

                                # Test
                                accuracy = test(model, device, test_loader)

                                # save models with acc > 60%
                                if accuracy >= 60:
                                    save_path = f"data/models/lenet5_opt_{optimizer.__name__}_lr_{lr}_reg_{reg}_epochs_{epochs}_seed_{seed}_init_{init}_batchsize_{batch_size}.pth"
                                    torch.save({
                                        'model_state_dict': model.state_dict(),
                                        'optimizer': optimizer.__name__,
                                        'learning_rate': lr,
                                        'regularization': reg,
                                        'epochs': epochs,
                                        'seed': seed,
                                        'initialization': init,
                                        'batch_size': batch_size,
                                        'accuracy': accuracy
                                    }, save_path)
                                # print(f"Saved model to {save_path} with accuracy {accuracy:.2f}%")
                                pbar.set_description(f"Optm: {optimizer.__name__}, Lr: {lr}, Reg: {reg}, Epochs: {epochs}, Seed: {seed}, Init: {init}, Batch size: {batch_size}, Accuracy: {accuracy:.2f}%")
                                pbar.update(1)
                                del model
                                torch.mps.empty_cache()

    pbar.close()


if __name__ == '__main__':
    main()