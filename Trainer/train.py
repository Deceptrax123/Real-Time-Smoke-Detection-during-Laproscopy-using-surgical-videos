import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from Trainer.video_dataset import VideoDataset
from torch import nn
from time import time
import torch.multiprocessing as mp
import wandb
from torch import mps, cpu
from sklearn.model_selection import train_test_split
import gc
import os
from dotenv import load_dotenv


if __name__ == '__main__':
    mp.set_sharing_strategy('file_system')
    load_dotenv('.env')

    # Video path
    videos_path = os.getenv("video_global_path")

    # test the dataset class
    dataset = VideoDataset(os.listdir(videos_path+"video05/"), "video05")

    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }

    loader = DataLoader(dataset, **params)

    # Test the loader
    for i, (X, y) in enumerate(loader):
        print(X.size())
