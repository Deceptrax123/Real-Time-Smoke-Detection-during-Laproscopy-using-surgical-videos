import os
from dotenv import load_dotenv
def count_frames():
    path = os.getenv("video_global_path")
    videos = os.listdir(path)
    for v in videos:
        video_path = os.path.join(path,v)
        samples = os.listdir(video_path)
        for s in samples:
            sample_path = os.path.join(video_path,s)
            frames = os.listdir(sample_path)
            if(len(frames) != 10):
                print(f"{sample_path} contains {len(frames)}")
                removable = [f for f in frames if "(" in f]
                print(f"Attempting to remove: {removable}")
                print(f"This will result in {len(frames) - len(removable)} images.")
                conf = input()
                if conf.lower() == 'y':
                    for p in removable:
                        os.remove(os.path.join(sample_path,p))



if __name__ == '__main__':
    load_dotenv(".env")
    count_frames()