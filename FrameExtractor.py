import datetime
import math
import os
import cv2
from tqdm import tqdm


def get_video_frames(path: str, every: int = 1) -> []:
    """
    Returns a list of the locations of the extracted frames.

    :param path: where the frames are saved.
    :param every: get every ith extracted frame.

    :returns images: a list of the locations of the frames
    """
    files = os.listdir(path)
    images = []
    if path[-1] != '/':
        path += '/'
    for i in tqdm(range(0, len(files), every)):
        frame = 'frame_' + str(i) + '.jpg'
        images.append(path + frame)

    return images


class FrameExtractor:
    """
    Class used for extracting frames from a video file.
    Adapted from:
    https://github.com/erykml/medium_articles/blob/master/Computer%20Vision/downloading_youtube_videos.ipynb
    """

    def __init__(self, video_path):
        assert video_path, 'Please enter a path!'
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        """Method for printing the video's duration"""
        duration = self.n_frames / self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame, verbose=True):
        """
        Method for calculating the expected number of images to save given
        we save every x-th frame

        Parameters
        ----------

        every_x_frame : int
            Indicates we want to look at every x-th frame
        verbose: bool
            Whether to print output message. Default is True.
        """
        n_images = math.floor(self.n_frames / every_x_frame) + 1

        if verbose:
            print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg'):
        """
        Method used for extracting the frames from images

        Parameters
        ----------

        every_x_frame : int
            Indicates we want to extract every x-th frame
        img_name : str
            The image name, numbers will be appended (after an underscore) at the end
        dest_path : str
            The path where to store the images. Default (None) saves the images to current directory.
        img_ext : str
            Indicates the desired extension of the image. Default is JPG
        """
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        pbar = tqdm(total=self.get_n_images(every_x_frame=every_x_frame, verbose=False))
        while self.vid_cap.isOpened():

            success, image = self.vid_cap.read()

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1
                pbar.update(every_x_frame)

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()
