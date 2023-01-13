import configparser
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
import cv2 as cv

from tensorflow.keras import backend as K


class GradCAM:
    """Grad-CAM uses gradient information from a convolutional layer to understand the importance of each neuron for a decision of interest.

    """

    def __init__(self, config_file):
        self.model = None
        self.model_hash = None
        self.layer_name = None
        # Reads the configuration file.
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def set_model(self, hash):
        """Selects the trained model of the experiment.

        :param model: Hash of the trained model.
        :type model:  :class:`str`
        """
        listing = os.listdir(os.path.join(self.config['PATHS']['experiments'], 'models'))

        for dir in listing:
            if hash in dir:
                self.model_hash = dir
                break

        model = tf.keras.models.load_model(os.path.join(self.config['PATHS']['experiments'], 'models', self.model_hash))
        self.model = model

    def set_layer_name(self, layer_name):
        """Selects the layer to be investigated by the Grad-CAM.

        :param layer_name: Convolutional layer name.
        :type layer_name:  :class:`str`
        """
        self.layer_name = layer_name

    def get_heatmap(self, img, class_id):
        """
        Calculate a heatmap using the Grad-CAM model.

        :param img: Input image.
        :type img: :class:`numpy.ndarray`
        :param class_id: True image label.
        :type class_id: :class:`tf.tensor`
        :returns: :class:`tuple` -- Tuple (heatmap, pred) where heatmap is the map calculated by Grad-CAM and pred is the prediction made by the model for the input image.
        """
        model_grad = tf.keras.Model([self.model.inputs],
                                    [self.model.get_layer(self.layer_name).output,
                                     self.model.output])

        # Gradient tape.
        with tf.GradientTape() as tape:
            conv_output_values, predictions = model_grad(img)
            loss = predictions[:, class_id]

        # Compute the gradients.
        grads_values = tape.gradient(loss, conv_output_values)
        # Mean of gradients for feature map.
        grads_values = K.mean(grads_values, axis=(0, 1, 2))
        conv_output_values = np.squeeze(conv_output_values.numpy())
        grads_values = grads_values.numpy()

        for i in range(len(grads_values)):
            conv_output_values[:, :, i] *= grads_values[i]

        # Heatmap.
        heatmap = np.mean(conv_output_values, axis=-1)
        # Remove negative values.
        heatmap = np.maximum(heatmap, 0)
        # Normalize.
        heatmap /= heatmap.max()
        pred = int(tf.argmax(predictions, axis=1, output_type=tf.int32))

        del model_grad, conv_output_values, grads_values, loss

        return heatmap, pred

    def show_sample(self, datagen, name, frame_idx):
        """Applies Grad-CAM to a frame of a video and displays the frame, heatmap and both superimposed.

        :param datagen: DataGen that has the video you want to investigate.
        :type datagen:  :class:`DataGen`
        :param name: Video name.
        :type name:  :class:`str`
        :param frame_idx: Frame index within the video.
        :type frame_idx:  :class:`int`
        """
        temp = datagen.batch_size
        datagen.batch_size = 1

        try:
            idx = datagen.df.index[datagen.df['name'] == name].tolist()
        except:
            raise Exception('{} is not in DataGen.'.format(name))

        try:
            X, y = datagen[idx[frame_idx]]
        except:
            raise Exception('Frame "{}" is not in video.'.format(frame_idx))

        y = tf.argmax(y[0], axis=0, output_type=tf.int32)

        heatmap, pred = self.get_heatmap(X, y)
        heatmap = cv.resize(heatmap, (X.shape[1], X.shape[2]))
        heatmap = heatmap * 255
        heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        heatmap = cv.applyColorMap(heatmap, cv.COLORMAP_VIRIDIS)

        super_imposed_image = cv.addWeighted(X[0], 0.8, heatmap.astype('float32'), 2e-3, 0.0, dtype=cv.CV_64F)

        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

        ax[0].imshow(X[0])
        ax[0].set_title("True label: {} \n Predicted label: {}".format(y, pred))
        ax[0].axis('off')

        ax[1].imshow(heatmap)
        ax[1].set_title("Class Activation Map")
        ax[1].axis('off')

        ax[2].imshow(super_imposed_image)
        ax[2].set_title("Activation map superimposed")
        ax[2].axis('off')
        plt.tight_layout()
        plt.show()

        datagen.batch_size = temp
