import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import random

def r_square(y_true, y_pred):
    """
    Calculate the r square of the model
    :param y_true:
    :param y_pred:
    :return: float between 0 and 1
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_mean = np.mean(y_true)
    ss_tot = np.sum(np.square(y_true - y_mean))
    ss_res = np.sum(np.square(y_true - y_pred))
    return 1 - ss_res / ss_tot

def test(model: nn.Module, dataloader: DataLoader, loss_fn, device: str = "mps") -> (float, float):
    """ 
        Test function, return the corresponding loss of the model on the given dataset. 
        This is a test function for regression models. Therefore, no acc will be returned.

    Args:
        model (nn.Module): the model to be tested
        dataloader (DataLoader): the dataset (can be a subset of the whole dataset)
        loss_fn (_label_type_): the label_type of loss function
        device (str, optional): device for pytorch. Defaults to "mps".

    Returns:
        float: the corresponding loss of the model on the given dataset
        float: the corresponding r square of the model on the given dataset
    """

    model = model.to(device)
    model.eval()
    loss = 0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_idx, (x_maxpool2, x_fc1, x_fc2, x_fc3, ratio, _, _, _) in enumerate(dataloader):
            x_maxpool2 = x_maxpool2.to(device)
            x_fc1 = x_fc1.to(device)
            x_fc2 = x_fc2.to(device)
            x_fc3 = x_fc3.to(device)
            ratio = ratio.to(device)
            output = model(x_maxpool2, x_fc1, x_fc2, x_fc3).squeeze(-1)
            loss += loss_fn(output, ratio).item()

            # Store all true and predicted values for R² calculation
            y_true_all.append(ratio.cpu())
            y_pred_all.append(output.cpu())

    # Concatenate all batches to calculate R²
    y_true_all = torch.cat(y_true_all, dim=0)
    y_pred_all = torch.cat(y_pred_all, dim=0)

    r_square_value = r_square(y_true_all, y_pred_all)

    return loss / len(dataloader), r_square_value


def test_cls(model: nn.Module, dataloader: DataLoader, loss_fn, device: str = "mps") -> float:
    """
        Test function, return the corresponding loss of the model on the given dataset.
        This is a test function for regression models. Therefore, no acc will be returned.

    Args:
        model (nn.Module): the model to be tested
        dataloader (DataLoader): the dataset (can be a subset of the whole dataset)
        loss_fn (_label_type_): the label_type of loss function
        device (str, optional): device for pytorch. Defaults to "mps".

    Returns:
        float: the corresponding loss of the model on the given dataset
    """

    model = model.to(device)
    model.eval()
    loss = 0
    total_num = 0
    correct_num = 0
    with torch.no_grad():
        for batch_idx, (x_maxpool2, x_fc1, x_fc2, x_fc3, _, label_type, _, _,) in enumerate(dataloader):
            x_maxpool2 = x_maxpool2.to(device)
            x_fc1 = x_fc1.to(device)
            x_fc2 = x_fc2.to(device)
            x_fc3 = x_fc3.to(device)
            label_type = label_type.to(device)
            output = model(x_maxpool2, x_fc1, x_fc2, x_fc3)
            loss += loss_fn(output, label_type).item()
            total_num += len(label_type)
            correct_num += (output.argmax(dim=1) == label_type).sum().item()


    return loss / len(dataloader), correct_num / total_num


def train(model: nn.Module, train_data: Dataset, test_data: Dataset, loss_fn, optm, epoch: int, batch_size: int = 32, device: str = "mps") -> (list[float], list[float]):
    """ 
        Train function, return the corresponding loss of the model on the given dataset. 
        This is a train function for regression models. Therefore, no acc will be returned.

    Args:
        model (nn.Module): the model to be trained
        train_data (torch.utils.data.Dataset): the training dataset 
        test_data (torch.utils.data.Dataset): the testing dataset 
        loss_fn (_label_type_): the label_type of loss function
        optm (_label_type_): the label_type of optimizer
        epoch (int): the number of epochs
        batch_size (int, optional): batch size for training. Defaults to 1000.
        device (str, optional): device for pytorch. Defaults to "mps".

    Returns:
        (list[float], list[float]): the corresponding loss of the model on the given dataset
    """

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    model = model.to(device)
    model.train()
    pbar = tqdm(total=epoch, leave=True, position=0, ncols=100)
    train_loss_list = []
    test_loss_list = []
    train_r_square_list = []
    test_r_square_list = []

    for epoch_idx in range(epoch):
        for batch_idx, (x_maxpool2, x_fc1, x_fc2, x_fc3, ratio, _, _, _) in enumerate(train_loader):
            x_maxpool2 = x_maxpool2.to(device)
            x_fc1 = x_fc1.to(device)
            x_fc2 = x_fc2.to(device)
            x_fc3 = x_fc3.to(device)
            ratio = ratio.to(device)
            optm.zero_grad()
            output = model(x_maxpool2, x_fc1, x_fc2, x_fc3).squeeze(-1)
            loss = loss_fn(output, ratio)
            loss.backward()
            optm.step()

        train_loss, train_r_square = test(model, dataloader=train_loader, loss_fn=loss_fn, device=device)
        test_loss, test_r_square = test(model, dataloader=test_loader, loss_fn=loss_fn, device=device)

        pbar.set_description(f"Epoch: {epoch_idx + 1} | Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}")
        pbar.update(1)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_r_square_list.append(train_r_square)
        test_r_square_list.append(test_r_square)
    pbar.close()

    return train_loss_list, test_loss_list, train_r_square_list, test_r_square_list


def train_cls(model: nn.Module, train_data: Dataset, test_data: Dataset, loss_fn, optm, epoch: int, batch_size: int = 32, device: str = "mps") -> (list[float], list[float]):
    """
        Train function, return the corresponding loss of the model on the given dataset.
        This is a train function for regression models. Therefore, no acc will be returned.

    Args:
        model (nn.Module): the model to be trained
        train_data (torch.utils.data.Dataset): the training dataset
        test_data (torch.utils.data.Dataset): the testing dataset
        loss_fn (_label_type_): the label_type of loss function
        optm (_label_type_): the label_type of optimizer
        epoch (int): the number of epochs
        batch_size (int, optional): batch size for training. Defaults to 1000.
        device (str, optional): device for pytorch. Defaults to "mps".

    Returns:
        (list[float], list[float]): the corresponding loss of the model on the given dataset
    """

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)
    model = model.to(device)
    model.train()
    pbar = tqdm(total=epoch, leave=True, position=0, ncols=100)
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch_idx in range(epoch):
        for batch_idx, data in enumerate(train_loader):
            (x_maxpool2, x_fc1, x_fc2, x_fc3, _, label_type, _, _) = data
            x_maxpool2 = x_maxpool2.to(device)
            x_fc1 = x_fc1.to(device)
            x_fc2 = x_fc2.to(device)
            x_fc3 = x_fc3.to(device)
            label_type = label_type.to(device)

            optm.zero_grad()
            output = model(x_maxpool2, x_fc1, x_fc2, x_fc3)
            loss = loss_fn(output, label_type)
            loss.backward()
            optm.step()

        train_loss, train_acc = test_cls(model, dataloader=train_loader, loss_fn=loss_fn, device=device)
        test_loss, test_acc = test_cls(model, dataloader=test_loader, loss_fn=loss_fn, device=device)

        pbar.set_description(f"Epoch: {epoch_idx + 1} | Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}")
        pbar.update(1)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    pbar.close()

    return train_loss_list, test_loss_list, train_acc_list, test_acc_list


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
