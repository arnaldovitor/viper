import configparser
import abc
import os
import tkinter

import cv2 as cv
import pandas as pd

from PIL import Image, ImageTk
from tkinter import messagebox


class Annotation(metaclass=abc.ABCMeta):
    """Dataset annotation class.

    """
    def __init__(self, config_file, dataset, file_name):
        # Reads the configuration file.
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.dataset = dataset
        self.file_name = file_name
        self.row = None
        self.idx = 0
        # Root.
        self.root = tkinter.Tk()
        self.root.title('Annotation')
        self.root.geometry('1920x1080')
        # Frame.
        self.frame = tkinter.Frame(self.root)
        self.frame.pack()
        # Display image.
        self.label = tkinter.Label(self.frame, compound=tkinter.TOP)
        self.label.pack()
        # Progress text.
        self.progress_str = tkinter.StringVar()
        self.progress = tkinter.Label(self.frame, compound=tkinter.TOP, textvariable=self.progress_str)
        self.progress.pack()
        # Current video name.
        self.video_name_str = tkinter.StringVar()
        self.video_name = tkinter.Label(self.frame, compound=tkinter.TOP, textvariable=self.video_name_str)
        self.video_name.pack()
        # Current video label.
        self.video_label_str = tkinter.StringVar()
        self.video_label = tkinter.Label(self.frame, compound=tkinter.TOP, textvariable=self.video_label_str)
        self.video_label.pack()
        # Input to a new label.
        self.new_video_label_str = tkinter.StringVar()
        self.new_video_label = tkinter.Entry(self.frame, textvariable=self.new_video_label_str)
        self.new_video_label.pack(side=tkinter.LEFT)
        # Update button.
        self.update = tkinter.Button(self.frame, text='Update', command=self.__update_label)
        self.update.pack(side=tkinter.LEFT)
        # Next and previous buttons.
        self.next = tkinter.Button(self.frame, text='>', command=lambda: self.__move(1))
        self.next.pack(side=tkinter.RIGHT)
        self.previous = tkinter.Button(self.frame, text='<', command=lambda: self.__move(-1))
        self.previous.pack(side=tkinter.RIGHT)

    def __count_frames(self, video_path):
        """Split and shuffle a DataFrame into partitions.

        :param video_path: Path to video.
        :type video_path: :class:`str`
        :returns: :class:`int` -- Number of frames in a video.

        """
        cap = cv.VideoCapture(video_path)
        frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frames

    def __capture_frame(self, video_path, index):
        """Returns the frame of a video indicated by an index in the correct format for Tkinter.

        :param video_path: Path to video.
        :type video_path: :class:`str`
        :param index: Index that indicates the position of the frame within the video.
        :type index: :class:`int`
        :returns: :class:`PIL.ImageTk.PhotoImage` -- Video frame.

        """
        cap = cv.VideoCapture(video_path)
        cap.set(1, index)
        _, frame = cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # Resizes the image if it is larger than the screen.
        if frame.shape[1] > 1280 or frame.shape[0] > 720:
            width = int(frame.shape[1] * 0.6)
            height = int(frame.shape[0] * 0.6)
            dim = (width, height)
            frame = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
        # CV Image -> Tk Image
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(image=frame)

        return frame

    @abc.abstractmethod
    def __setup(self):
         pass

    @abc.abstractmethod
    def __move(self, delta):
         pass

    @abc.abstractmethod
    def __update_label(self):
        pass

    @abc.abstractmethod
    def run(self):
        pass

class FramewiseAnnotation(Annotation):
    """Annotation child class for framewise annotation.

    """
    def __init__(self, config_file, dataset, file_name, inter_sequence_skip=None):
        super().__init__(config_file, dataset, file_name)
        self.inter_sequence_skip = inter_sequence_skip
        self.annotated_frames = pd.DataFrame(columns=['name', 'index', 'label'])
        # Setup.
        self._Annotation__setup()

    def _Annotation__setup(self):
        """It defines some initial settings for the annotation object, such as: reading or creating an annotation file and defining the first image and texts to be shown.

        """
        # Exception handling.
        if self.dataset not in ['vf', 'hockey', 'ucf_indoor', 'rwf_indoor', 'rwf_2000', 'aie']:
            raise Exception('{} it is not a dataset supported by the framework.'.format(self.dataset))
        if not os.path.exists(os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset))):
            raise Exception('{} needs to be downloaded.'.format(self.dataset))

        # Creates or reads an annotation file.
        if not os.path.exists(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name))):
            dataset_dir = os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), self.dataset)
            listing = os.listdir(dataset_dir)
            for video in listing:
                video_path = os.path.join(dataset_dir, video)
                frames = self._Annotation__count_frames(video_path)
                for i in range(0, frames, self.inter_sequence_skip):
                    new_row = pd.DataFrame([{'name': video, 'index': i, 'label': '?'}])
                    self.annotated_frames = pd.concat([self.annotated_frames, new_row])
            self.annotated_frames.to_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)))
        else:
            self.annotated_frames = pd.read_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)))

        # Curret row on df.
        self.row = self.annotated_frames.iloc[self.idx]
        img = self._Annotation__capture_frame(os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), self.dataset, self.row['name']), self.row['index'])
        # Update image and texts.
        self.label.imgtk = img
        self.label.configure(image=img)
        self.video_name_str.set('Name: {}'.format(self.row['name']))
        self.video_label_str.set('Label: {}'.format(self.row['label']))
        self.progress_str.set('{}/{}'.format(self.idx + 1, len(self.annotated_frames)))

    def _Annotation__move(self, delta):
        """Returns the frame of a video indicated by an index.

        :param delta: Value used to update the current row of the DataFrame.
        :type delta: :class:`int`
        """
        if not 0 <= self.idx + delta < len(self.annotated_frames):
            messagebox.showinfo('End', 'No more image')
        else:
            self.idx += delta
            # Curret row on df.
            self.row = self.annotated_frames.iloc[self.idx]
            # CV image to Tk image.
            img = self._Annotation__capture_frame(os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), self.dataset, self.row['name']), self.row['index'])
            # Update image and texts.
            self.label.imgtk = img
            self.label.configure(image=img)
            self.video_name_str.set('Name: {}'.format(self.row['name']))
            self.video_label_str.set('Label: {}'.format(self.row['label']))
            self.progress_str.set('{}/{}'.format(self.idx + 1, len(self.annotated_frames)))

    def _Annotation__update_label(self):
        """Updates the current image label.

        """
        new_label = self.new_video_label_str.get()
        self.annotated_frames.iloc[self.idx, [3]] = new_label
        self.annotated_frames.to_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)), index=False)
        self.video_label_str.set('Label: {}'.format(new_label))


class SequentialAnnotation(Annotation):
    """Annotation child class for sequential annotation.

    """
    def __init__(self, config_file, dataset, file_name, inter_sequence_skip=None, intra_sequence_skip=None, sequence_size=None):
        super().__init__(config_file, dataset, file_name)
        self.inter_sequence_skip = inter_sequence_skip
        self.intra_sequence_skip = intra_sequence_skip
        self.sequence_size = sequence_size
        self.annotated_frames = pd.DataFrame(columns=['name', 'index', 'label', 'intra', 'size'])
        # Setup.
        self._Annotation__setup()

    def __play_sequence(self, i=0):
        """Displays a sequence of video frames in the GUI.

        :param i: Value used to walk between frames in a sequence.
        :type i: :class:`int`
        """
        if i >= self.sequence_size:
            i = 0
        img = self._Annotation__capture_frame(os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), self.dataset, self.row['name']), int(self.row['index']) + (i*self.intra_sequence_skip))
        # Update image and texts.
        self.label.imgtk = img
        self.label.configure(image=img)
        self.root.after(1, self.__play_sequence, i + 1)

    def _Annotation__setup(self):
        """It defines some initial settings for the annotation object, such as: reading or creating an annotation file and defining the first image and texts to be shown.

        """
        if self.dataset not in ['vf', 'hockey', 'ucf_indoor', 'rwf_indoor', 'rwf_2000', 'aie']:
            raise Exception('{} it is not a dataset supported by the framework.'.format(self.dataset))
        if not os.path.exists(os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset))):
            raise Exception('{} needs to be downloaded.'.format(self.dataset))

        # Creates or reads an annotation file.
        if not os.path.exists(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name))):
            dataset_dir = os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), self.dataset)
            listing = os.listdir(dataset_dir)
            for video in listing:
                video_path = os.path.join(dataset_dir, video)
                frames = self._Annotation__count_frames(video_path)
                for i in range(0, frames-(self.intra_sequence_skip*(self.sequence_size-1)), (self.intra_sequence_skip*(self.sequence_size-1))+self.inter_sequence_skip):
                    new_row = pd.DataFrame([{'name': video, 'index': i, 'label': '?', 'intra': self.intra_sequence_skip, 'size': self.sequence_size}])
                    self.annotated_frames = pd.concat([self.annotated_frames, new_row])
                self.annotated_frames.to_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)))
        else:
            self.annotated_frames = pd.read_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)))
            tmp = self.annotated_frames.iloc[0]
            self.sequence_size = tmp['size']
            self.intra_sequence_skip = tmp['intra']

        # Curret row on df.
        self.row = self.annotated_frames.iloc[self.idx]
        self.video_name_str.set('Name: {}'.format(self.row['name']))
        self.video_label_str.set('Label: {}'.format(self.row['label']))
        self.progress_str.set('{}/{}'.format(self.idx + 1, len(self.annotated_frames)))

    def _Annotation__move(self, delta):
        """Returns the frame of a video indicated by an index.

        :param delta: Value used to update the current row of the DataFrame..
        :type delta: :class:`int`
        """
        if not 0 <= self.idx + delta < len(self.annotated_frames):
            messagebox.showinfo('End', 'No more image')
        else:
            self.idx += delta
            # Curret row on df.
            self.row = self.annotated_frames.iloc[self.idx]
            self.video_name_str.set('Name: {}'.format(self.row['name']))
            self.video_label_str.set('Label: {}'.format(self.row['label']))
            self.progress_str.set('{}/{}'.format(self.idx + 1, len(self.annotated_frames)))

    def _Annotation__update_label(self):
        """Updates the current image label.

        """
        new_label = self.new_video_label_str.get()
        self.annotated_frames.iloc[self.idx, [3]] = new_label
        self.annotated_frames.to_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, self.file_name)), index=False)
        self.video_label_str.set('Label: {}'.format(new_label))

    def run(self):
        """Starts the framewise annotation interface.

        """
        self.__play_sequence()
        self.root.mainloop()