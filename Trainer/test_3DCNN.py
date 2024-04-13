import torch
from Trainer.video_dataset import VideoDatasetTest
from torch.utils.data import DataLoader
from Trainer.Models.CNN_3D.model import Conv3DBase
import os
import argparse
import pandas as pd





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='sum the integers at the command line')
    parser.add_argument('--mode', type=str, choices=["all","per_video","per_sample"], default="per_sample", required=True,
        help='Mode in which to run inference')
    parser.add_argument('--path', type=str,required=False,
        help='Path to folder')
    parser.add_argument('--save_path', type=str,required=False,default="out.csv",
        help='Path to folder')
    args = parser.parse_args()
    model = Conv3DBase().to(device="cuda")
    model.load_state_dict(torch.load("Trainer/weights/run_2/model21.pth"))
    model.eval()
    if(args.mode == "all"):
        pred = {}
        if(args.path is not None):
            videos = [x for x in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, x))]
            total = len(videos)
            for j,v in enumerate(videos):
                video_path = os.path.join(args.path, v)
                samples = [x for x in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, x))]
                for sample in samples :
                    if len(os.listdir(os.path.join(video_path, sample))) != 10 :
                        raise AssertionError(f"Number of samples in {os.path.join(video_path, sample)} not equal to 10")
    
                dl = VideoDatasetTest(paths=[os.path.join(video_path, s) for s in samples])
                loader=DataLoader(dl,**{
                    "batch_size":8,
                    "shuffle":True,
                    "num_workers":0
                })
                for path, x in loader:
                    x : torch.Tensor = x.to(device="cuda")
                    _, predictions = model(x)
                    predictions = predictions > 0.5
                    for i in range(len(path)):
                        if bool(predictions[i]):
                            pred["_".join(path[i].split("\\")[-2:]).split("/")[-1]] = 1
                        else:
                            pred["_".join(path[i].split("\\")[-2:]).split("/")[-1]] = 0
                df = pd.DataFrame.from_records(zip(list(pred.keys()),list(pred.values())),columns=('sample_name','label'))
                df.to_csv(args.save_path, index=False)
                print(f"Processed {j}/{total}")
                
                
        else:
            raise argparse.ArgumentError("Missing folder path in mode: all. Use --path to pass this.")
    elif(args.mode == "per_video"):
        if(args.path is not None):
            samples = [x for x in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, x))]
            for video_path in samples : 
                if len(os.listdir(os.path.join(args.path, video_path))) != 10 :
                    raise AssertionError(f"Number of samples in {os.path.join(args.path, video_path)} not equal to 10")
            dl = VideoDatasetTest(paths=[os.path.join(args.path, s) for s in samples])
            
            loader=DataLoader(dl,**{
                "batch_size":8,
                "shuffle":True,
                "num_workers":0
            })
            pred = {}
            for path, x in loader:
                x : torch.Tensor = x.to(device="cuda")
                _, predictions = model(x)
                # predictions = predictions > 0.5
                for i in range(len(path)):
                    pred[path[i]] = float(predictions[i])
            for k in pred.keys():
                print("_".join(k.split("/")[-2:]),pred[k], pred[k] > 0.5)
        else:
            raise argparse.ArgumentError("Missing folder path in mode: all. Use --path to pass this.")
    else:
        if(args.path is not None):
            if len(os.listdir(args.path)) != 10 :
                raise AssertionError(f"Number of samples in {args.path} not equal to 10")
            dl = VideoDatasetTest(paths=[args.path,])
            
            loader=DataLoader(dl,**{
                "batch_size":8,
                "shuffle":False,
                "num_workers":0
            })
            pred = {}
            for path, x in loader:
                x : torch.Tensor = x.to(device="cuda")
                _, predictions = model(x)
                predictions = predictions > 0.5
                for i in range(len(path)):
                    pred[path[i]] = bool(predictions[i])
            for k in pred.keys():
                print("_".join(k.split("/")[-2:]),pred[k], pred[k] > 0.5)
                    
                
        else:
            raise argparse.ArgumentError("Missing folder path in mode: all. Use --path to pass this.")
    
    
    
    