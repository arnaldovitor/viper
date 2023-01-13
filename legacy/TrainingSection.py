import tensorflow as tf
import abc
import os
import random
import configparser
# import yagmail
import shutil

from tensorflow.python.keras.utils.generic_utils import Progbar


class TrainingSection(metaclass=abc.ABCMeta):
    """Object that represents a model training experiment.

    """
    def __init__(self, config_file):
        self.train_data = None
        self.validation_data = None
        self.model = None
        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.train_acc = None
        self.validation_acc = None
        self.epochs = None
        # self.reports = False
        # self.reports_email = None
        # self.reports_interval = None
        self.train_hash = str(random.getrandbits(128))
        self.history = {'best-train-acc': 0.0,
                        'best-val-acc': 0.0,
                        'best-train-epoch': 0,
                        'best-val-epoch': 0}
        # Reads the configuration file.
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def set_train_data(self, train_data):
        """Selects the DataGen used to train the model of the experiment.

        :param train_data: Training DataGen.
        :type train_data:  :class:`DataGen`
        """
        self.train_data = train_data

    def set_validation_data(self, validation_data):
        """Selects the DataGen used to validate the model of the experiment.

        :param validation_data: Validation DataGen.
        :type validation_data:  :class:`DataGen`
        """
        self.validation_data = validation_data

    def set_model(self, model):
        """Selects the model of the experiment.

        :param model: TensorFlow model.
        :type model:  :class:`tf.keras.Sequential`
        """
        self.model = model

    # def enable_reports(self, email, interval):
        # self.reports = True
        # self.reports_email = email
        # self.reports_interval = interval

    def __save_model(self):
        """Save trained model in directory selected in configuration file.

        """
        if not os.path.exists(self.config['PATHS']['experiments']):
            os.mkdir(self.config['PATHS']['experiments'])
        if not os.path.exists(os.path.join(self.config['PATHS']['experiments'], 'models')):
            os.mkdir(os.path.join(self.config['PATHS']['experiments'], 'models'))
        # if os.path.exists(os.path.join(self.config['PATHS']['experiments'],'models', '{}_{}'.format(self.model.name, self.train_hash))):
        #     shutil.rmtree(os.path.join(self.config['PATHS']['experiments'],'models', '{}_{}'.format(self.model.name, self.train_hash)))
        self.model.save(os.path.join(self.config['PATHS']['experiments'],'models', '{}_{}'.format(self.model.name, self.train_hash)))


    # def __send_email(self, subject, contents):
        # yag = yagmail.SMTP(user='', password='')
        # yag.send(to=self.reports_email, subject=subject, contents=contents)

    @abc.abstractmethod
    def __train_step(self, x, y):
        pass

    @abc.abstractmethod
    def __val_step(self, x, y):
        pass

    @abc.abstractmethod
    def run(self, optimizer, loss, train_acc, validation_acc, epochs, save_condition=None):
        pass


class ClassificationTrainingSection(TrainingSection):
    """Child class of TrainingSection, used for classification experiments.

    """
    def __init__(self, config_file):
        super().__init__(config_file)

    @tf.function
    def _TrainingSection__train_step(self, x, y):
        """
        Private method that computes one step of the Backpropagation technique to update the network weights during the
        training stage.

        :param x: 4D tensor with dimensions: [batch, height, width, channel] composed of frames of a video.
        :type x: :class:`tf.tensor`
        :param y: Tensor that represents the target output relative to an x of the training set.
        :type y: :class:`tf.tensor`
        :returns: :class:`tf.tensor` -- Error after one step of the Backpropagation technique.
        """

        with tf.GradientTape() as tape:
            pred = self.model(x, training=True)
            err = self.loss(y, pred)

        grads = tape.gradient(err, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_acc.update_state(y, pred)
        return err

    @tf.function
    def _TrainingSection__val_step(self, x, y):
        """
        Private method that evaluates the model's performance against a validation set.

        :param x: 4D tensor with dimensions: [batch, height, width, channel] composed of frames of a video.
        :type x: :class:`tf.tensor`
        :param y: Tensor that represents the target output relative to an x of the training set.
        :type y: :class:`tf.tensor`
        :returns: :class:`tf.tensor` -- Error in validation set.
        """
        pred = self.model(x, training=False)
        err = self.loss(y, pred)
        self.validation_acc.update_state(y, pred)
        return err

    def run(self, optimizer, loss, train_acc, validation_acc, epochs, save_condition = None):
        """
        Sets hypermeters and runs the training.

        :param optimizer: Optimizer instance.
        :type optimizer: :class:`tf.keras.optimizers`
        :param loss: Loss instance.
        :type loss: :class:`tf.keras.losses`
        :param train_acc: Train accuracy instance.
        :type train_acc: :class:`tf.keras.metrics`
        :param validation_acc: Validation accuracy instance.
        :type validation_acc: :class:`tf.keras.metrics`
        :param epochs: Number of epochs for training.
        :type epochs: :class:`int`
        :param save_condition: Condition to save the model during training.
        :type save_condition: :class:`str`
        """

        self.optimizer = optimizer
        self.loss = loss
        self.train_acc = train_acc
        self.validation_acc = validation_acc

        for epoch in range(epochs):
            print("\nTRAINING: epoch {}/{}".format(epoch + 1, epochs))
            pb_train = Progbar(len(self.train_data) * self.train_data.batch_size, stateful_metrics=['train-loss', 'train-acc'])

            total_train_loss = 0.0
            for i, (x, y) in enumerate(self.train_data):
                x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                total_train_loss += self._TrainingSection__train_step(x_tensor, y_tensor)
                if i % 2 == 0:
                    pb_train.update(i * self.train_data.batch_size, values=[('train-loss', total_train_loss / len(self.train_data)), ('train-acc', self.train_acc.result())])

            print("\nVALIDATING: epoch {}/{}".format(epoch + 1, epochs))
            pb_val = Progbar(len(self.validation_data) * self.validation_data.batch_size, stateful_metrics=['val-loss', 'val-acc'])

            total_val_loss = 0.0
            for i, (x, y) in enumerate(self.validation_data):
                x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
                total_val_loss += self._TrainingSection__val_step(x_tensor, y_tensor)
                if i % 2 == 0:
                    pb_val.update(i * self.validation_data.batch_size, values=[('val-loss', total_val_loss / len(self.validation_data)), ('val-acc', self.validation_acc.result())])

            # Updates dictionary with better accuracies and saves the model according to the condition passed to the run function.
            if self.train_acc.result() > self.history['best-train-acc']:
                self.history['best-train-acc'] = float(self.train_acc.result())
                self.history['best-train-epoch'] = epoch+1
                if save_condition == 'best_train':
                    self._TrainingSection__save_model()
            if self.validation_acc.result() > self.history['best-val-acc']:
                self.history['best-val-acc'] = float(self.validation_acc.result())
                self.history['best-val-epoch'] = epoch+1
                if save_condition == 'best_val':
                    self._TrainingSection__save_model()
            if save_condition is None:
                self._TrainingSection__save_model()

            # Send e-mail with report.
            # if self.reports and (epoch+1) % self.reports_interval == 0:
                # subject = '[TRAINING REPORT] Model: {}, Epoch: {}/{}'.format(self.model.name, epoch+1, epochs)
                # contents = 'Epoch: {}\n\ntrain-loss: {}\ntrain-acc: {}\n\nval-loss: {}\nval-acc: {}'.format(epoch+1,
                                                                                                          # total_train_loss / len(self.train_data),
                                                                                                          # self.train_acc.result(),
                                                                                                          # total_val_loss / len(self.validation_data),
                                                                                                          # self.validation_acc.result())

                # self._TrainingSection__send_email(subject, contents)

            self.train_acc.reset_states()
            self.validation_acc.reset_states()

            self.train_data.on_epoch_end()
            self.validation_data.on_epoch_end()

        return self.history

