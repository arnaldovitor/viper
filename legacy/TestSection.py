import json
import os
import abc
import configparser

import tensorflow as tf

from tensorflow.python.keras.utils.generic_utils import Progbar
from DataGen import DataGenFramewiseClassification


class TestSection(metaclass=abc.ABCMeta):
    """Object that represents a model testing experiment.

    """
    def __init__(self, config_file):
        self.testing_data = None
        self.model = None
        self.model_hash = None
        self.loss = None
        self.test_acc = None
        # Reads the configuration file.
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def set_testing_data(self, testing_data):
        """Selects the DataGen used to test the model of the experiment.

        :param testing_data: Training DataGen.
        :type testing_data:  :class:`DataGen`
        """
        self.testing_data = testing_data

    def set_model(self, hash):
        """Selects the trained model of the experiment.

        :param hash: Hash of the trained model..
        :type hash:  :class:`str`
        """
        listing = os.listdir(os.path.join(self.config['PATHS']['experiments'], 'models'))

        for dir in listing:
            if hash in dir:
                self.model_hash = dir
                break

        model = tf.keras.models.load_model(os.path.join(self.config['PATHS']['experiments'], 'models', self.model_hash))
        self.model = model

    @abc.abstractmethod
    def __test_step(self, x, y):
        pass

    @abc.abstractmethod
    def run(self, name, loss, test_acc):
        pass


class ClassificationTestSection(TestSection):
    """Child class of TestSection, used for classification experiments.

    """
    def __init__(self, config_file):
        super().__init__(config_file)

    @tf.function
    def _TestSection__test_step(self, x, y):
        """
        Private method that evaluates the model's performance against a test set.

        :param x: 4D tensor with dimensions: [batch, height, width, channel] composed of frames of a video.
        :type x: :class:`tf.tensor`
        :param y: Tensor that represents the target output relative to an x of the training set.
        :type y: :class:`tf.tensor`
        :returns: :class:`tuple` -- Tuple (pred, err) where pred is the predictions made by the model and err is error after one step of the Backpropagation technique
        """
        pred = self.model(x, training=False)
        err = self.loss(y, pred)
        self.test_acc.update_state(y, pred)
        return pred, err

    def run(self, name, loss, test_acc):
        """
        Sets hypermeters and runs the training.

        :param loss: Loss instance.
        :type loss: :class:`tf.keras.losses`
        :param test_acc: Test accuracy instance.
        :type test_acc: :class:`tf.keras.metrics`
        """

        self.loss = loss
        self.test_acc = test_acc

        if name != 'all':
            # Removes DataFrame columns that do not match the video selected for testing.
            df = self.testing_data.df.drop(self.testing_data.df[name != self.testing_data.df['name']].index)

            # Creates a new DataGen based on the reduced DataFrame.
            datagen = DataGenFramewiseClassification(df,
                                                     batch_size=1,
                                                     input_size=self.testing_data.input_size,
                                                     id_label=self.testing_data.id_label,
                                                     n_label=self.testing_data.n_label)
        else:
            datagen = self.testing_data

        total_test_loss = 0.0
        # Empty dictionary to save predictions.
        partitions = {k: {'label': -1, 'predictions': []} for k in datagen.df['name']}

        print('\nTESTING:')
        for i, (x, y) in enumerate(datagen):
            pb_test = Progbar(len(datagen) * datagen.batch_size, stateful_metrics=['train-loss', 'train-acc'])
            x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
            y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
            pred, err = self._TestSection__test_step(x_tensor, y_tensor)

            # Loop under the predictions of a batch.
            for j, temp in enumerate(pred):
                # Organize dimensions of an individual prediction.
                temp = tf.expand_dims(temp, axis=0)
                # One-hot-encoding -> label-encoding
                temp = int(tf.argmax(temp, axis=1, output_type=tf.int32))
                # Saves the true label in the dictionary if not saved.
                if partitions[datagen.df['name'].iloc[j + (i*datagen.batch_size)]]['label'] == -1:
                    partitions[datagen.df['name'].iloc[j + (i*datagen.batch_size)]]['label'] = int(tf.argmax(tf.expand_dims(y[j], axis=0), axis=1, output_type=tf.int32))
                # Associates a frame's prediction to the video it belongs to.
                partitions[datagen.df['name'].iloc[j + (i*datagen.batch_size)]]['predictions'].append(temp)

            total_test_loss += err
            pb_test.update(i * datagen.batch_size, values=[('test-loss', total_test_loss / len(datagen)), ('test-acc', self.test_acc.result())])


        if not os.path.exists(os.path.join(self.config['PATHS']['experiments'], 'predictions')):
            os.mkdir(os.path.join(self.config['PATHS']['experiments'], 'predictions'))

        with open(os.path.join(self.config['PATHS']['experiments'], 'predictions', 'framewise_{}_{}.json'.format(self.model_hash, name)), 'w') as out_file:
            json.dump(partitions, out_file, indent=2)

    def accuracy_by_vote(self, pred_file):
        """
        Calculates accuracy using voting after a test experiment.

        :param pred_file: Name of the JSON file generated after a test experiment.
        :type pred_file: :class:`tf.keras.losses`
        """

        with open(os.path.join(self.config['PATHS']['experiments'], 'predictions', pred_file)) as json_file:
            pred_dict = json.load(json_file)

        correct_predictions = 0

        for key in pred_dict:
            video = pred_dict[key]
            pred = max(set(video['predictions']), key=video['predictions'].count)

            print('{} (true/pred): {}/{}'.format(key, video['label'], pred))
            if pred == video['label']:
                correct_predictions+=1

        print('\nCorrect predictions: {}'.format(correct_predictions))
        print('Accuracy by vote:  {}'.format(correct_predictions/len(pred_dict)))