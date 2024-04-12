import torch
import torchvision.transforms as T
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from torch.utils.data import DataLoader
from Trainer.video_dataset import VideoDataset
from torch import nn
from time import time
import torch.multiprocessing as mp
import wandb
from torch import mps, cpu
from Trainer.Models.CNN_3D.model import Conv3DBase
from sklearn.model_selection import train_test_split
import gc
import os
from dotenv import load_dotenv


def train_step(train_loader):
    epoch_loss = 0
    epoch_precision = 0
    epoch_accuracy = 0
    epoch_recall = 0

    for step, (x_sample, label) in enumerate(train_loader):
        x_sample = x_sample.to(device=device)
        label = label.to(device=device)

        # Forward Pass
        logits, predictions = model(x_sample)

        model.zero_grad()

        # Compute Loss
        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, label)

        loss.backward()
        model_optimizer.step()

        epoch_loss += loss.item()

        # Move tensors to cpu
        x_sample = x_sample.to(device='cpu')
        label = label.to(device='cpu')
        predictions = predictions.to(device='cpu')
        logits = logits.to(device='cpu')

        # Metrics
        epoch_precision += precision(predictions, label).item()
        epoch_recall += recall(predictions, label).item()
        epoch_accuracy += accuracy(predictions, label).item()

        # Memory Handling
        mps.empty_cache()

        # step-wise inferencing
        print("Step: ", step)
        print("Step Train Loss: ", loss.item())
        print("Step Train Accuracy: ", accuracy(predictions, label).item())
        print("Step Train Precision: ", precision(predictions, label).item())
        print("Step Train Recall", recall(predictions, label).item())

        wandb.log({
            "Step Train Loss": loss.item(),
            "Step Train Accuracy": accuracy(predictions, label).item(),
            "Step Train Precision": precision(predictions, label).item(),
            "Step Train Recall": recall(predictions, label).item(),

        })

        del x_sample
        del label
        del predictions
        del logits

    return epoch_loss/(step+1), epoch_accuracy/(step+1), epoch_precision/(step+1), epoch_recall/(step+1)


def test_step(test_loader):
    epoch_loss = 0
    epoch_precision = 0
    epoch_accuracy = 0
    epoch_recall = 0

    for step, (x_sample, label) in enumerate(test_loader):
        x_sample = x_sample.to(device=device)
        label = label.to(device=device)

        # Forward Pass
        logits, predictions = model(x_sample)

        loss_function = nn.BCEWithLogitsLoss()
        loss = loss_function(logits, label)

        # Move tensors to CPU
        x_sample = x_sample.to(device='cpu')
        label = label.to(device='cpu')
        predictions = predictions.to(device='cpu')
        logits = logits.to(device='cpu')

        epoch_loss += loss.item()

        # Metrics
        epoch_precision += precision(predictions, label).item()
        epoch_recall += recall(predictions, label).item()
        epoch_accuracy += accuracy(predictions, label).item()

        # step-wise inferencing
        print("Step: ", step)
        print("Step Test Loss: ", loss.item())
        print("Step Test Accuracy: ", accuracy(predictions, label).item())
        print("Step Test Precision: ", precision(predictions, label).item())
        print("Step Test Recall", recall(predictions, label).item())

        wandb.log({
            "Step Test Loss": loss.item(),
            "Step Test Accuracy": accuracy(predictions, label).item(),
            "Step Test Precision": precision(predictions, label).item(),
            "Step Test Recall": recall(predictions, label).item(),

        })

        # Memory Handling
        mps.empty_cache()

        del x_sample
        del label
        del predictions
        del logits

    return epoch_loss/(step+1), epoch_accuracy/(step+1), epoch_precision/(step+1), epoch_recall/(step+1)


def training_loop():
    # For all Videos(Global Perspective)

    for epoch in range(num_epochs):
        model.train(True)

        etrain_loss = 0
        etest_loss = 0
        etrain_prec = 0
        etest_prec = 0
        etrain_accuracy = 0
        etest_accuracy = 0
        etrain_rec = 0
        etest_rec = 0

        print("Values for Epoch: ", epoch)
        for video_id, video in enumerate(os.listdir(videos_path)):
            # Perspective->Each Video

            # Segment wise train-test split

            segments = os.listdir(videos_path+video+"/")
            train, test = train_test_split(segments, test_size=0.20)

            train_dataset = VideoDataset(train, video)
            test_dataset = VideoDataset(test, video)

            train_loader = DataLoader(train_dataset, **params)
            test_loader = DataLoader(test_dataset, **params)

            print("Video Number: ", video_id)

            model.train(True)

            train_loss, train_accuracy, train_prec, train_rec = train_step(
                train_loader)

            etrain_loss += train_loss
            etrain_accuracy += train_accuracy
            etrain_prec += train_prec
            etrain_rec += train_rec

            model.eval()

            with torch.no_grad():
                test_loss, test_accuracy, test_prec, test_rec = test_step(
                    test_loader)

                etest_loss += test_loss
                etest_accuracy += test_accuracy
                etest_prec += test_prec
                etest_rec += test_rec

        print("Overall Results")
        print("Train Loss: ", etrain_loss/60)
        print("Train Precision: ", etrain_prec/60)
        print("Train Recall: ", etrain_rec/60)
        print("Train Accuracy: ", etrain_accuracy/60)

        print("Test Loss: ", etest_loss/60)
        print("Test Precision: ", etest_prec/60)
        print("Test Accuracy: ", etest_accuracy/60)
        print("Test Recall: ", etest_rec/60)

        # Log to weights and biases
        wandb.log({
            "Train Loss": etrain_loss,
            "Test Loss": etest_loss,
            "Train Accuracy": etrain_accuracy,
            "Train Precision": etrain_prec,
            "Train Recall": etrain_rec,

            "Test Accuracy": etest_accuracy,
            "Test Precision": etest_prec,
            "Test Recall": etest_rec
        })


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

    wandb.init(
        project="Stryker Hackathon",
        config={
            "architecture": "Convolution Based Models",
            "Dataset": "Stryker",
        }
    )

    LR = 0.001
    num_epochs = 100

    # Model
    device = torch.device("cpu")
    model = Conv3DBase().to(device=device)

    # Optimizer
    model_optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999))

    # metrics
    precision = BinaryPrecision()
    recall = BinaryRecall()
    accuracy = BinaryAccuracy()

    training_loop()
