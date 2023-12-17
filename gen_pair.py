import torch
import numpy as np
from LeNet5 import LeNet5, LeNet5_obs
import os
from tqdm import tqdm

source_dir = "data/model_sample"
target_dir = "data/pair_9_60"

np_target_dir = "data/pair_9_60_np"
downsample_target_dir = "data/pair_9_npy"
scale = 0.9

def gen_random(scale=0.1, randome_seed=42, repeat=100):
    torch.manual_seed(randome_seed)
    return torch.randn(repeat, 1, 28, 28)*scale

def noise_sample(noise, sample, repeat=100):
    sample.unsqueeze(0).repeat(repeat, 1, 1, 1)
    return sample + noise

def gen_label(model:torch.nn.Module, sample:torch.tensor, prediction:int)->float:
    """
    get the confidence of the model in response to the sample
    Args:
        model: the model to be tested
        sample: the sample to be tested
        prediction: the original prediction of the sample

    Returns:
        the ratio of the unchanged prediction
    """
    pred = model(sample).argmax(dim=1)
    return (pred == prediction).sum().item()/len(pred)

def gen_pair(source_dir, target_dir, device="mps", repeat=1000, scale=0.1):
    ckpt = torch.load(source_dir)
    weights = ckpt["model_state_dict"]
    right_samples = ckpt["right_samples"]
    wrong_samples = ckpt["wrong_samples"]
    nosie = gen_random(repeat=repeat, scale=scale)
    
    model = LeNet5()
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    
    ratio = []
    type = []
    weight = {
        "conv1": torch.cat([weights["conv1.weight"].view(6, -1), weights["conv1.bias"].unsqueeze(1)], dim=1),
        "conv2": torch.cat([weights["conv2.weight"].view(16, -1), weights["conv2.bias"].unsqueeze(1)], dim=1),
        "fc1": torch.cat([weights["fc1.weight"], weights["fc1.bias"].unsqueeze(1)], dim=1),
        "fc2": torch.cat([weights["fc2.weight"], weights["fc2.bias"].unsqueeze(1)], dim=1),
        "fc3": torch.cat([weights["fc3.weight"], weights["fc3.bias"].unsqueeze(1)], dim=1)
    }
    samples = []
    labels_ground = []
    
    for i in range(10):
        if right_samples[i]:
            for sample in right_samples[i]:
                noised_sample = noise_sample(nosie, sample, repeat=repeat).to(device)
                ratio.append(gen_label(model, noised_sample, i))
                type.append(True)
                labels_ground.append(i)
                
            samples += right_samples[i]
            
        if wrong_samples[i]:
            for sample in wrong_samples[i]:
                noised_sample = noise_sample(nosie, sample, repeat=repeat).to(device)
                sample = sample.to(device)
                pred = model(sample).argmax(dim=1).item()
                ratio.append(gen_label(model, noised_sample, pred))
                type.append(False)
                labels_ground.append(i)
            samples += wrong_samples[i]
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    else:
        pass
    samples = torch.stack(samples).to(device)
    model = LeNet5_obs()
    model.load_state_dict(weights)
    model.to(device)
    model.eval()
    
    output = model(samples)
    
    labels_pred = output["fc3"].argmax(dim=1).squeeze().cpu().tolist()
    sample_seq = {
        "conv1": output["conv1"],
        "maxpool1": output["maxpool1"],
        "conv2": output["conv2"],
        "maxpool2": output["maxpool2"],
        "fc1": output["fc1"],
        "fc2": output["fc2"],
        "fc3": output["fc3"]
    } 
    torch.save((weight, samples, labels_ground, labels_pred, sample_seq, ratio, type), target_dir)

def load_data(file_path:str):
    weight, samples, ground_label, pred_label, sample_seq, ratio, type = torch.load(file_path)
    num_pairs = len(ratio)
    ratio = torch.tensor(ratio)
    type = torch.tensor(type)
    ground_label = torch.tensor(ground_label)
    pred_label = torch.tensor(pred_label)
    
    conv1 = weight["conv1"].repeat(num_pairs, 1, 1)
    conv2 = weight["conv2"].repeat(num_pairs, 1, 1)
    fc1 = weight["fc1"].repeat(num_pairs, 1, 1)
    fc2 = weight["fc2"].repeat(num_pairs, 1, 1)
    fc3 = weight["fc3"].repeat(num_pairs, 1, 1)
    
    x_conv1 = sample_seq["conv1"].flatten(start_dim=-2)
    x_maxpool1 = sample_seq["maxpool1"].flatten(start_dim=-2)
    x_conv2 = sample_seq["conv2"].flatten(start_dim=-2)
    x_maxpool2 = sample_seq["maxpool2"].flatten(start_dim=-2)
    x_fc1 = sample_seq["fc1"]
    x_fc2 = sample_seq["fc2"]
    x_fc3 = sample_seq["fc3"]
    
    return conv1, conv2, fc1, fc2, fc3, x_conv1, x_maxpool1, x_conv2, x_maxpool2, x_fc1, x_fc2, x_fc3, ratio, type, ground_label, pred_label

def main():
    files = os.listdir(source_dir)

    # generate pair data
    for f in tqdm(files):
        source_dir_file = os.path.join(source_dir, f)
        target_dir_file = os.path.join(target_dir, f)
        gen_pair(source_dir_file, target_dir_file, device="mps", repeat=1000, scale=scale)
    
    # generate npy data for last four layers' output to save memory
    files = os.listdir(target_dir)

    x_fc1_l = []
    x_fc2_l = []
    x_fc3_l = []
    x_maxpool2_l = []
    ratio_l = []
    type_l = []
    ground_label_l = []
    pred_label_l = []

    for file in tqdm(files):
        conv1, conv2, fc1, fc2, fc3, x_conv1, x_maxpool1, x_conv2, x_maxpool2, x_fc1, x_fc2, x_fc3, ratio, type, ground_label, pred_label = load_data(f"{par_dir}/{file}")
        x_fc1_l.append(x_fc1.cpu().detach().numpy())
        x_fc2_l.append(x_fc2.cpu().detach().numpy())
        x_fc3_l.append(x_fc3.cpu().detach().numpy())
        x_maxpool2_l.append(x_maxpool2.cpu().detach().numpy())
        ratio_l.append(ratio.cpu().detach().numpy())
        type_l.append(type.cpu().detach().numpy())
        ground_label_l.append(ground_label.cpu().detach().numpy())
        pred_label_l.append(pred_label.cpu().detach().numpy())
        
    x_fc1 = np.concatenate(x_fc1_l, axis=0)
    x_fc2 = np.concatenate(x_fc2_l, axis=0)
    x_fc3 = np.concatenate(x_fc3_l, axis=0)
    x_maxpool2 = np.concatenate(x_maxpool2_l, axis=0)
    total_num = x_fc1.shape[0]

    # save to npy
    np.save(f"{np_target_dir}/x_fc1.npy", x_fc1)
    np.save(f"{np_target_dir}/x_fc2.npy", x_fc2)
    np.save(f"{np_target_dir}/x_fc3.npy", x_fc3)
    np.save(f"{np_target_dir}/x_maxpool2.npy", x_maxpool2)

    # generate downsampled data
    # the size of the whole dataset is about 800,000, which is too large to fit in memory
    # so we downsample the dataset to 10% of the original size
    np.random.seed(0)
    sampled_idx = np.random.choice(total_num, total_num//10, replace=False)
    files = ['ground_label.npy',
            'pred_label.npy',
            'ratio.npy',
            'type.npy',
            'x_fc1.npy',
            'x_fc2.npy',
            'x_fc3.npy',
            'x_maxpool2.npy']

    for f in tqdm(files):
        data = np.load(f"{np_target_dir}/{f}")
        sampled_data = data[sampled_idx]
        np.save(f"{downsample_target_dir}/{f}", sampled_data)

if __name__ == "__main__":
    main()
    