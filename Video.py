import os

import validators
from utils import download_youtube_video
from FrameExtractor import FrameExtractor, get_video_frames


class Video:
    """
    This class creates a dataset for a Valorant gameplay video, starting from downloading it ending with outputting a
    .csv file.

    __init__ Parameters
    -----------

    every_x_frame: every ith frame of the video is extracted. Ideally, should be the framerate.

    Attributes
    ---------

    """

    def __init__(self, vid_location: str, save_dir: str, filename: str = None, every_x_frame: int = 30):
        assert vid_location is not None, 'Please enter a vid_location!'
        assert save_dir is not None, 'Please enter a save_dir'
        self.vid_location = vid_location
        self.save_dir = save_dir
        self.every_x_frame = every_x_frame
        self.filename = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if validators.url(vid_location):  # if the video is an url
            if filename[-4:] != '.mp4':
                filename += '.mp4'
            self.filename = filename

            if not os.path.exists(os.path.join(save_dir, f'{self.filename[:-4]}_frames')):
                os.makedirs(os.path.join(save_dir, f'{self.filename[:-4]}_frames'))

            download_youtube_video(url=vid_location, path=save_dir, filename=filename)
            self.vid_dir = os.path.join(save_dir, filename)
        else:
            assert filename is None, f"If vid_location is not a URL, filename has to be empty. Got '{filename}'!"
            if self.vid_location[-4:] != '.mp4':
                self.vid_location += '.mp4'

            self.vid_dir = self.vid_location

        self.frames = self.extract_frames()

    def extract_frames(self):
        """
        Extracts the frames from the video

        returns
        ---------
        frames: a list of the location of all the extracted frames
        """
        print("Extracting frames!")

        fe = FrameExtractor(self.vid_dir)
        if self.filename is None:
            path = os.path.join(self.save_dir, f"{self.vid_location[:-4].split('/')[-1]}_frames")
        else:
            path = os.path.join(self.save_dir, f'{self.filename[:-4]}_frames')

        if not os.path.exists(path):
            os.makedirs(path)

        if len(os.listdir(path)) == 0:
            fe.extract_frames(every_x_frame=self.every_x_frame, img_name='frame', dest_path=path)

        frames = get_video_frames(path)
        print("Frames extracted!")

        return frames
