import torch
import torchvision
import torchvision.transforms as T
from preprocessing.preprocessing import cropToCentreAdaptive
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, paths, video_id):
        self.paths = paths
        self.video_id = video_id

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Gets a sample
        load_dotenv(".env")

        global_video_path = os.getenv("video_global_path")
        sample_path = self.paths[index]

        # Get The Frames
        frame_path_list = os.listdir(
            global_video_path+self.video_id+"/"+sample_path+"/")

        # Read each of the 10 frames and stack the frames
        frames = list()
        for frame in frame_path_list:
            path = global_video_path+self.video_id+"/"+sample_path+"/"+frame

            img = Image.open(path)

            # Adaptive cropping algorithm
            img = cropToCentreAdaptive(
                np.array(img), threshold=0.90, step_size=8)

            # perform general pre-processing
            transforms = T.Compose(
                [T.ToTensor(), T.Normalize(mean=(0, 0, 0,), std=(1, 1, 1))])

            # Transform to Tensor
            img_tensor = transforms(img)

            # Stack the Frames together
            frames.append(img_tensor)

        X = torch.stack(frames, dim=1)

        # Get the Label
        if int(sample_path[-1]) == 1:
            label = torch.ones(1)
        else:
            label = torch.zeros(1)

        return X, label
