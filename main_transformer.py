import torch
from torch.utils.data import  DataLoader, random_split

import argparse
import os
import logging
import numpy as np
import random
import pandas as pd

from dataset import PairDataset_Signal
from train import train, test, train_cls, test_cls

from PredictConfidenceNet import ConfidenceNet_transformer, ConfidenceNet_transformer_cls

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Train the model to predict the confidence of LeNet5 given specific intput.')
    # arguments for training
    parser.add_argument("--cls", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--norm_first", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--decoder_num_layers", type=int, default=2)
    parser.add_argument("--decoder_hidden_size", type=int, default=256)
    parser.add_argument("--decoder_act", type=str, default='relu')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument("--target_dir", type=str, default=".", help="target directory to save the model")
    parser.add_argument("--data_dir", type=str, default=".", help="data directory")

    args = parser.parse_args()

    # set logger
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # set random seed
    setup_seed(args.random_seed)
    logging.info(f"Random seed: {args.random_seed}")

    # load data
    logging.info("Loading data...")

    x_maxpool2 = torch.from_numpy(np.load(f"{args.data_dir}/x_maxpool2.npy"))
    x_maxpool2 = x_maxpool2.reshape(x_maxpool2.shape[0], -1)
    x_fc1 = torch.from_numpy(np.load(f"{args.data_dir}/x_fc1.npy"))
    x_fc2 = torch.from_numpy(np.load(f"{args.data_dir}/x_fc2.npy"))
    x_fc3 = torch.from_numpy(np.load(f"{args.data_dir}/x_fc3.npy"))

    ratio = torch.from_numpy(np.load(f"{args.data_dir}/ratio.npy"))
    type = torch.from_numpy(np.load(f"{args.data_dir}/type.npy").astype(np.int64))
    ground_label = torch.from_numpy(np.load(f"{args.data_dir}/ground_label.npy"))
    pred_label = torch.from_numpy(np.load(f"{args.data_dir}/pred_label.npy"))

    logging.info(f"shape of x_maxpool2: {x_maxpool2.shape}")
    logging.info(f"shape of x_fc1: {x_fc1.shape}")
    logging.info(f"shape of x_fc2: {x_fc2.shape}")
    logging.info(f"shape of x_fc3: {x_fc3.shape}")

    whole_dataset = PairDataset_Signal(x_maxpool2, x_fc1, x_fc2, x_fc3, ratio, type, ground_label, pred_label)
    logging.info(f"data loading finished.")

    # split dataset
    logging.info("Spliting dataset...")
    train_dataset, test_dataset = random_split(whole_dataset, [int(len(whole_dataset) * 0.8), len(whole_dataset) - int(len(whole_dataset) * 0.8)])
    train_dataset, val_dataset = random_split(train_dataset, [int(len(train_dataset) * 0.8), len(train_dataset) - int(len(train_dataset) * 0.8)])
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    logging.info(f"train dataset size: {len(train_dataset)}")
    logging.info(f"val dataset size: {len(val_dataset)}")
    logging.info(f"test dataset size: {len(test_dataset)}")
    logging.info(f"dataset spliting finished.")

    # initialize the model
    if args.cls:
        model = ConfidenceNet_transformer_cls(embed_dim=args.embed_dim,
                                          nhead=args.nhead,
                                          dim_feedforward=args.dim_feedforward,
                                          num_transformer_layers=args.num_encoder_layers,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          layer_norm_eps=args.layer_norm_eps,
                                          norm_first=args.norm_first,
                                          bias=args.bias,
                                          decoder_num_layers=args.decoder_num_layers,
                                          decoder_hidden_size=args.decoder_hidden_size,
                                          decoder_act=args.decoder_act)
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        model = ConfidenceNet_transformer(embed_dim=args.embed_dim,
                                          nhead=args.nhead,
                                          dim_feedforward=args.dim_feedforward,
                                          num_transformer_layers=args.num_encoder_layers,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          layer_norm_eps=args.layer_norm_eps,
                                          norm_first=args.norm_first,
                                          bias=args.bias,
                                          decoder_num_layers=args.decoder_num_layers,
                                          decoder_hidden_size=args.decoder_hidden_size,
                                          decoder_act=args.decoder_act)
        loss_fn = torch.nn.MSELoss()
    if args.optimizer == "adam":
        optm = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optm = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}, only support adam, sgd")
    logging.info(f"model, optimizer and loss function initialization finished.")

    # train the model
    logging.info("Start training...")
    if args.cls:
        train_loss, val_loss, train_acc, val_acc = train_cls(model=model, train_data=train_dataset, test_data=val_dataset, loss_fn=loss_fn, optm=optm, epoch=args.epoch, batch_size=args.batch_size, device=args.device)
    else:
        train_loss, val_loss, train_r_square, val_r_square = train(model=model, train_data=train_dataset, test_data=val_dataset, loss_fn=loss_fn, optm=optm, epoch=args.epoch, batch_size=args.batch_size, device=args.device)
    logging.info("Training finished.")
    logging.info("Testing model...")
    if args.cls:
        test_loss, test_acc = test_cls(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=args.device)
        logging.info(f"Test loss: {test_loss}, Test acc: {test_acc}")
    else:
        test_loss, test_r_square = test(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=args.device)
        logging.info(f"Test loss: {test_loss}, Test r_square: {test_r_square}")
    logging.info("Testing finished.")
    logging.info("saving model...")
    model_name = f"model_transformer_cls_{args.random_seed}.pth" if args.cls else f"model_transformer_reg_{args.random_seed}.pth"
    torch.save(model.state_dict(), f"{args.target_dir}/{model_name}")
    logging.info("model saved.")

    logging.info("saving results")
    if args.cls:
        df = pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})
        df.to_csv(f"{args.target_dir}/results_transformer_cls_{args.random_seed}.csv")
    else:
        df = pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss, "train_r_square": train_r_square, "val_r_square": val_r_square})
        df.to_csv(f"{args.target_dir}/results_transformer_{args.random_seed}.csv")
    logging.info("results saved.")

if __name__ == "__main__":
    main()





