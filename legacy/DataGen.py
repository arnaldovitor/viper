import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv
import abc
from matplotlib import pyplot as plt

class DataGen(tf.keras.utils.Sequence, metaclass=abc.ABCMeta):
    """Custom object to train TensorFlow models.

    """

    def __init__(self, df, batch_size, input_size, id_label, n_label, crop, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.id_label = id_label
        self.n_label = n_label
        self.crop = crop
        self.shuffle = shuffle

        self.n = len(self.df)

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
        """Returns the frame of a video indicated by an index.

        :param video_path: Path to video.
        :type video_path: :class:`str`
        :param index: Index that indicates the position of the frame within the video.
        :type index: :class:`int`
        :returns: :class:`numpy.ndarray` -- Video frame.

        """
        cap = cv.VideoCapture(video_path)
        cap.set(1, index)
        _, frame = cap.read()

        return frame

    def __rgb_to_flow(self, frame, next_frame):
        """Converts a video frame to its representation in Dense Optical Flow.

        :param frame: Video frame.
        :type frame: :class:`numpy.ndarray`
        :param next_frame: Frame adjacent to the converted.
        :type next_frame: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- Representation in Dense Optical Flow.

        """
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        next_frame = cv.cvtColor(next_frame, cv.COLOR_BGR2GRAY)
        # Calculate optical flow between each pair of frame;
        flow = cv.calcOpticalFlowFarneback(frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, cv.OPTFLOW_FARNEBACK_GAUSSIAN)
        # Subtract the mean in order to eliminate the movement of camera.
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        # Normalize each component in optical flow.
        flow[..., 0] = cv.normalize(flow[..., 0], None, 0, 255, cv.NORM_MINMAX)
        flow[..., 1] = cv.normalize(flow[..., 1], None, 0, 255, cv.NORM_MINMAX)

        return flow

    def __dynamic_crop(self, frame, flow):
        # Distance between the center and end of the crop.
        dist_x = int(self.input_size[0]/2)
        dist_y = int(self.input_size[1]/2)
        # Get an Optical Flow component.
        magnitude = flow[..., 1]
        # Filter slight noise by threshold.
        thresh = np.mean(magnitude)
        magnitude[magnitude < thresh] = 0
        # Calculate center of gravity of magnitude map and adding 0.001 to avoid empty value.
        x_pdf = np.sum(magnitude, axis=1) + 0.001
        y_pdf = np.sum(magnitude, axis=0) + 0.001
        # Normalize PDF of x and y so that the sum of probs = 1.
        x_pdf /= np.sum(x_pdf)
        y_pdf /= np.sum(y_pdf)
        # Randomly choose some candidates for x and y.
        x_points = np.random.choice(a=np.arange(frame.shape[0]), size=10, replace=True, p=x_pdf)
        y_points = np.random.choice(a=np.arange(frame.shape[1]), size=10, replace=True, p=y_pdf)
        # Get the mean of x and y coordinates for better robustness.
        x = int(np.mean(x_points))
        y = int(np.mean(y_points))
        # Avoid to beyond boundaries of array.
        x = max(dist_x, min(x, frame.shape[0] - dist_x))
        y = max(dist_y, min(y, frame.shape[1] - dist_y))
        # Get cropped frame and flow.
        cropped_frame = frame[x - dist_x:x + dist_x, y - dist_y:y + dist_y, :]
        cropped_flow = flow[x - dist_x:x + dist_x, y - dist_y:y + dist_y, :]

        return cropped_frame, cropped_flow

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch.

         """
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path, modes, index, target_size):
        """Converts a video frame to its representation in Dense Optical Flow.

        :param path: Path to video.
        :type path: :class:`str`
        :param modes: List of frame representation modes used for extraction.
        :type modes:  :class:`list`
        :param index: Index that indicates the position of the frame within the video.
        :type index: :class:`int`
        :param target_size: Dimension used to resize the frame.
        :type target_size:  :class:`tuple`
        :returns: :class:`list` -- Array with the channels that represent the image according to the chosen mode.

        """
        all_arr = []

        frame_rgb = self.__capture_frame(path, index)

        if not self.crop:
            frame_rgb = cv.resize(frame_rgb, (target_size[0], target_size[1]))

            if 'frame_rgb' in modes:
                frame_rgb_arr = frame_rgb / 255.
                all_arr.append(frame_rgb_arr)
            if 'frame_gray' in modes:
                frame_gray = cv.cvtColor(frame_rgb, cv.COLOR_BGR2GRAY)
                frame_gray_arr = np.expand_dims(frame_gray, axis=-1)
                frame_gray_arr = frame_gray_arr / 255.
                all_arr.append(frame_gray_arr)
            if 'flow' in modes:
                next_frame_rgb = self.__capture_frame(path, index + 1)
                next_frame_rgb = cv.resize(next_frame_rgb, (target_size[0], target_size[1]))
                flow = self.__rgb_to_flow(frame_rgb, next_frame_rgb)
                flow_arr = flow / 255.
                all_arr.append(flow_arr)
        else:
            next_frame_rgb = self.__capture_frame(path, index + 1)
            flow = self.__rgb_to_flow(frame_rgb, next_frame_rgb)
            frame_rgb_cropped, flow_cropped = self.__dynamic_crop(frame_rgb, flow)

            if 'frame_rgb' in modes:
                frame_rgb_cropped_arr = frame_rgb_cropped / 255.
                all_arr.append(frame_rgb_cropped_arr)
            if 'frame_gray' in modes:
                frame_gray_croppped = cv.cvtColor(frame_rgb_cropped, cv.COLOR_BGR2GRAY)
                frame_gray_cropped_arr = np.expand_dims(frame_gray_croppped, axis=-1)
                frame_gray_cropped_arr = frame_gray_cropped_arr / 255.
                all_arr.append(frame_gray_cropped_arr)
            if 'flow' in modes:
                flow_cropped_arr = flow_cropped / 255.
                all_arr.append(flow_cropped_arr)

        comb_arr = np.concatenate(all_arr, axis=-1)

        return comb_arr

    def __get_output(self, label, num_classes):
        """Converts a class vector to binary class matrix.

        :param label: Current frame class.
        :type label: :class:`int`
        :param num_classes: Total number of classes in the dataset.
        :type num_classes: :class:`int`
        :returns: :class:`numpy.ndarray` -- One-hot encoded NumPy array.

        """
        return tf.keras.utils.to_categorical(label, num_classes=num_classes)

    @abc.abstractmethod
    def __get_data(self, batches):
        """Helper function that takes a batch and iterates over it calling other helper functions.

        :param batches: DataFrame representing a split batch..
        :type batches: :class:`DataFrame`
        :returns: :class:`tuple` -- Tuple (X, y) where X are the channels of all images and y their classes.

        """
        pass

    def __getitem__(self, index):
        """Get a batch of data from the DataFrame using indexing.

        :param index: Index indicating a split batch..
        :type index: :class:`int`
        :returns: :class:`tuple` -- Tuple (X, y) where X are the image channels and y the class.

        """
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(batches)

        return X, y


    def __len__(self):
        """Calculates the number of generated batches.

        :returns: :class:`int` -- Number of batches.

        """
        return -1 * (-self.n // self.batch_size)

class DataGenFramewiseClassification(DataGen):
    """Child class of DataGen, used for frame-by-frame classification of a set of videos.

    """

    def __init__(self, df, batch_size, input_size, id_label, n_label, crop):
        super().__init__(df, batch_size, input_size, id_label, n_label, crop)


    def _DataGen__get_data(self, batches):
        path_batch = batches['path']
        modes_batch = batches['modes']
        index_batch = batches['index']

        # Convert class type from string to int.
        label_batch = pd.Categorical(batches['label'], categories=self.id_label, ordered=True).codes

        X_batch = np.asarray([self._DataGen__get_input(x, y, z, self.input_size) for x, y, z in
                              zip(path_batch, modes_batch, index_batch)])

        y_batch = np.asarray([self._DataGen__get_output(y, self.n_label) for y in label_batch])

        return X_batch, y_batch


class DataGenSequentialClassification(DataGen):
    """Child class of DataGen, used for sequential classification of a set of videos.

    """
    def __init__(self, df, batch_size, input_size, id_label, n_label, sequence_size, intra_sequence_skip, crop):
        super().__init__(df, batch_size, input_size, id_label, n_label, crop)
        self.sequence_size = sequence_size
        self.intra_sequence_skip = intra_sequence_skip

    def _DataGen__get_data(self, batches):
        path_batch = batches['path']
        modes_batch = batches['modes']
        index_batch = batches['index']

        channel_size = 0
        if 'frame_rgb' in modes_batch.iloc[0]:
            channel_size+=3
        if 'frame_gray' in modes_batch.iloc[0]:
            channel_size+=1
        if 'flow' in modes_batch.iloc[0]:
            channel_size+=2

        # Convert class type from string to int.
        label_batch = pd.Categorical(batches['label'], categories=self.id_label, ordered=True).codes

        X_batch = np.zeros((len(batches), self.input_size[0], self.input_size[1], self.sequence_size, channel_size))
        y_batch = np.asarray([self._DataGen__get_output(y, self.n_label) for y in label_batch])

        for idx, (x, y, z) in enumerate(zip(path_batch, modes_batch, index_batch)):
            for n in range(self.sequence_size):
                X_batch[idx, :, :, n, :] = self._DataGen__get_input(x, y, z + (n*self.intra_sequence_skip), self.input_size)

        return X_batch, y_batch
