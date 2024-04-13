import cv2
from cv2 import VideoWriter
import os
import random
import torch
from dotenv import load_dotenv
from Trainer.video_dataset import VideoDatasetTest
from Trainer.ffmpegWriter import FFMPEGWriter
from torch.utils.data import DataLoader
from Trainer.Models.CNN_3D.model import Conv3DBase
def choose(n : int, choose_from_same_sample : bool = False, label : int = None):
    for i in range(n):
        path_to_videos = os.environ["video_global_path"]
        videos = [f for f in os.listdir(path_to_videos) if os.path.isdir(os.path.join(path_to_videos,f))]
        ch_video = random.choice(videos)
        path_to_samples = os.path.join(path_to_videos,ch_video)
        samples = os.listdir(path_to_samples)
        if(label is not None):
            samples = [s for s in samples if s.endswith(str(label))]
        ch_sample = random.choice(samples)
        path_to_sample = os.path.join(path_to_samples,ch_sample)
        yield path_to_sample
            

if __name__ == '__main__':
    load_dotenv()
    result_dir = os.getenv("demo_result_path")
    model = Conv3DBase().to(device="cuda")
    model.load_state_dict(torch.load(os.getenv("best_model_path")))
    model.eval()
    smoke_samples = list(choose(5,label=1))
    non_smoke_samples = list(choose(5,label=0))
    final = smoke_samples + non_smoke_samples
    
    dl = VideoDatasetTest(paths=final)
    loader=DataLoader(dl,**{
        "batch_size":8,
        "shuffle":True,
        "num_workers":0
    })
    pred = {}
    for path, x in loader:
        x : torch.Tensor = x.to(device="cuda")
        _, predictions = model(x)
        predictions = predictions > 0.5
        for i in range(len(path)):
            if bool(predictions[i]):
                pred[path[i]] = "Smoke"
            else:
                pred[path[i]] = "No Smoke"
    for i,path in enumerate(pred.keys()):
        short_name = f"video_{i}"
        frames = [os.path.join(path,x) for x in os.listdir(path)]
        print(frames)
        frames = [cv2.imread(p) for p in frames]
        os.mkdir(os.path.join(result_dir,short_name))
        save_path =os.path.join(result_dir,short_name)
        for j,f in enumerate(frames):
            f = cv2.putText(f, pred[path], (50,50), cv2.FONT_HERSHEY_SIMPLEX ,   
                   1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_path,f"{j}.png"),f)
        
        
        
        
        
        
