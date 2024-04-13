import cv2
import numpy as np
import subprocess as sp
import shlex

class FFMPEGWriter():
    def __init__(self, shape : np.ndarray, fps : float, output_filename = "output.mp4") -> None:
        self.width, self.height = shape
        self.output_filename = output_filename
        self.fps = fps
        self.process = sp.Popen(shlex.split(f'ffmpeg -y -s {self.width}x{self.height} -pixel_format bgr24 -f rawvideo -r {self.fps} -i pipe: -vcodec libx265 -pix_fmt yuv420p -crf 24 "{self.output_filename}" '), stdin=sp.PIPE)
    def write(self, frame : cv2.Mat):
        frame = np.array(frame,dtype=np.uint8).tobytes()
        self.process.stdin.write(frame)
    def release(self):
        self.process.stdin.close()
        self.process.wait()
        self.process.terminate()
    def __del__(self):
        self.release()
        