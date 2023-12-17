import torch
import pandas as pd
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

from model.LeNet5 import LeNet5

def load_model(model, model_path):
    """
    load model from .pth file
    :param model: pre-defined model object
    :param model_path: file path of .pth file
    :return: a model object with loaded weights
    """
    ckpt = torch.load(model_path, map_location=torch.device('mps'))
    return model.load_state_dict(ckpt['model_state_dict'])


def select_sample(model, test_dataloader, device):
    """
    select samples from test dataset
        1. select 200 samples for each model
        2. 100 samples with right predictions, 10 samples per class
        3. 100 samples with wrong predictions, 10 samples per class
    :param model: a model object with loaded weights
    :param test_dataloader: a dataloader object where samples come from
    :param device: torch device
    :return:
        two dictionaries, each contains 100 samples with right/wrong predictions
    """
    model.to(device)
    model.eval()
    right_samples = {i: [] for i in range(10)}
    wrong_samples = {i: [] for i in range(10)}

    for i, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for j in range(len(labels)):
            if len(right_samples[labels[j].item()]) < 10 and predicted[j] == labels[j]:
                right_samples[labels[j].to("cpu").item()].append(images[j].to("cpu"))
            elif len(wrong_samples[labels[j].item()]) < 10 and predicted[j] != labels[j]:
                wrong_samples[labels[j].to("cpu").item()].append(images[j].to("cpu"))

        # check if we have collected enough samples, enough means that
        #   in right samples, we have ten samples for each class
        #   in wrong samples, we have ten samples for each class
        if all(len(right_samples[i]) == 10 for i in range(10)) and all(len(wrong_samples[i]) == 10 for i in range(10)):
            break
    # check again, this time, if we finally do not have enough samples, we will raise a warning
    # if not all(len(right_samples[i]) == 10 for i in range(10)) or not all(len(wrong_samples[i]) == 10 for i in range(10)):
    #     warnings.warn("We do not have enough samples for each class, please check the dataset")
    return right_samples, wrong_samples


def save_model_sample(model, right_samples, wrong_samples, model_path):
    """
    save model and samples to .pth file
    Args:
        model: A model object with loaded weights
        right_samples: A dictionary contains 100 samples with right predictions
        wrong_samples: A dictionary contains 100 samples with wrong predictions
        model_path: the path to save the model and samples

    Returns:
        None
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'right_samples': right_samples,
        'wrong_samples': wrong_samples
    }, model_path)


def save_all(parent_dic, model_detail_path, model_parent_path, test_dataloader, device):
    """
    save all models and samples to .pth files in parent_dic
    Args:
        parent_dic: where to save the models and samples
        model_detail_path: a csv file contains the training details to all models
        model_parent_path: the parent path of all models
        test_dataloader: test dataloader
        device: torch device

    Returns:
        None
    """
    tqdm.pandas(desc="save models and samples")
    def process_fun(row):
        reg = row['regularization']
        if reg == 0:
            reg = int(reg)
        model_name = f"lenet5_opt_{row['optimizer']}_lr_{row['learning_rate']}_reg_{reg}_epochs_{row['epochs']}_seed_{row['seed']}_init_{row['initialization']}_batchsize_{row['batch_size']}.pth"
        model_path = f"{model_parent_path}/{model_name}"

        # check if the model exists
        if os.path.isfile(model_path):
            model = LeNet5()
            load_model(model, model_path)
            right_samples, wrong_samples = select_sample(model, test_dataloader, device)
            save_model_sample(model, right_samples, wrong_samples, f"{parent_dic}/{model_name}")
            del model
            torch.mps.empty_cache()

    # load model details and retrieve model names
    model_details = pd.read_csv(model_detail_path)
    model_details.progress_apply(process_fun, axis=1)

def main():
    parent_dic = "model_sample"
    model_detail_path = "model_info.csv"
    model_parent_path = "models"
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    device = "mps"  # change device if the machine is not MAC with M-series chip

    save_all(parent_dic, model_detail_path, model_parent_path, test_loader, device)

if __name__ == "__main__":
    main()