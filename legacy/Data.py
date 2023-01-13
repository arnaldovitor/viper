import pandas as pd
import cv2 as cv
import os
import json
import random
import configparser
import gdown
import zipfile

from DataGen import DataGenFramewiseClassification, DataGenSequentialClassification


class Data:
    """Object representing a dataset.

    To be instantiated, it needs a configuration file.
    """
    def __init__(self, config_file):
        self.dataset = None
        self.paper = None
        self.release_year = None
        self.resource = None
        self.hours_length = None
        self.number_of_instances = None
        self.classes_distribution = None
        self.dataset_dir = None
        self.splits_dir = None
        # Reads the configuration file.
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

    def __count_frames(self, video_path):
        """Split and shuffle a DataFrame into partitions.

        :param video_path: Path to video.
        :type video_path: :class:`str`
        :returns: :class:`int` -- Number of frames in a video.

        """
        cap = cv.VideoCapture(video_path)
        frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        return frames

    def __split_by_props(self, df, props, random_state=42):
        """Split and shuffle a DataFrame into partitions.

        :param df: DataFrame with dataset video names and classes.
        :type df: :class:`DataFrame`
        :param props: List with the proportions used to split.
        :type props:  :class:`list`
        :param random_state: Seed for random number generator.
        :type random_state: :class:`int`
        :returns: :class:`DataFrame` -- the splited DataFrame.

        """
        remain = df.index.copy().to_frame()
        res = []
        for i in range(len(props)):
            props_sum = sum(props[i:])
            frac = props[i] / props_sum
            idxs = remain.sample(frac=frac, random_state=random_state).index
            remain = remain.drop(idxs)
            res.append(idxs)
        return [df.loc[idxs] for idxs in res]


    def load(self, dataset):
        """Load the selected dataset.

        :param dataset: Dataset identifier.
        :type dataset: :class:`str`

        """
        if dataset not in ['vf', 'hockey', 'ucf_indoor', 'rwf_indoor', 'rwf_2000', 'aie']:
            raise Exception('{} it is not a dataset supported by the framework.'.format(dataset))

        self.dataset = dataset
        self.dataset_dir = os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset))
        self.splits_dir = os.path.join(self.config['PATHS']['datasets'], '{}_dir'.format(self.dataset), 'splits')

        if self.dataset == 'vf':
            self.paper = 'Violent Flows: Real-Time Detection of Violent Crowd Behavior'
            self.release_year = '2012'
            self.resource = 'Real-world outdoor'
            self.hours_length = '0.8'
            self.number_of_instances = '246'
            self.classes_distribution = {'violent': '123', 'non-violent': '123'}
            zip_file_id = '1mvjvlKkZ5d6ZmM88EOsOP1KOx6gCuX76'
            csv_file_id = '12JXyjytN2XJ9D_4nDwUEIaX2mmM4PO95'
        elif self.dataset == 'hockey':
            self.paper = 'Violence detection in video using computer vision techniques'
            self.release_year = '2012'
            self.resource = 'Sports'
            self.hours_length = '0.44'
            self.number_of_instances = '1000'
            self.classes_distribution = {'fight': '500', 'non-fight': '500'}
            zip_file_id = '14pprUzxFkx4roUHi2DeEVEhB3QwHZq0w'
            csv_file_id = '11XugJEVonk4H9h2JVBI2iUkWzVn1vjQd'
        elif self.dataset == 'ucf_indoor':
            self.paper = ''
            self.release_year = ''
            self.resource = 'Real-world indoor'
            self.hours_length = '5.55'
            self.number_of_instances = '2355'
            self.classes_distribution = {'violent': '517', 'non-violent': '1762', 'alert': '76'}
            zip_file_id = '1qccg9iKQMYB4F-DcFz0rYJvOP0BDegbw'
            csv_file_id = '1HOb-7UwKWECuLl03HxeOu3YbcucfvEfy'
        elif self.dataset == 'rwf_2000':
            self.paper = 'RWF-2000: An Open Large Scale Video Database for Violence Detection'
            self.release_year = '2020'
            self.resource = 'Real-world indoor and outdoor'
            self.hours_length = '2.8'
            self.number_of_instances = '2000'
            self.classes_distribution = {'violent': '1000', 'non-violent': '1000'}
            zip_file_id = '1_7PbGi25vPWPjSgGH77A-ha6-vIkiBYm'
            csv_file_id = '1bulmr1-Rov9TBTf5CwAMyjaDJ9VITNgm'
        elif self.dataset == 'rwf_indoor':
            self.paper = ''
            self.release_year = ''
            self.resource = 'Real-world indoor'
            self.hours_length = '1.19'
            self.number_of_instances = '870'
            self.classes_distribution = {'violent': '456', 'non-violent': '414'}
            zip_file_id = '1UwdCPc39gcQvYU5bN5hNQZmLXWGbYM8e'
            csv_file_id = '10THUJIGsi0-HVB1pCdHg5xsUPJos-yni'
        elif self.dataset == 'aie':
            self.paper = ''
            self.release_year = ''
            self.resource = 'Real-world indoor'
            self.hours_length = '1.83'
            self.number_of_instances = '700'
            self.classes_distribution = {'violent': '348', 'non-violent': '300', 'alert': '52'}
            zip_file_id = '1BE9d2Eyy44Ndn4t6OJwi5aBLXYgVCp91'
            csv_file_id = '1CQtDDR9IAUrtnGkId4SKEB7iF-RJMJrB'

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        if not os.path.exists(self.splits_dir):
            os.mkdir(self.splits_dir)

        if not os.path.exists(os.path.join(self.dataset_dir, '{}.zip'.format(self.dataset))):
            # Download dataset zip.
            gdown.download(id=zip_file_id,
                           output=os.path.join(self.dataset_dir, '{}.zip'.format(self.dataset)))

            print('Unziping...')

            # Unzip dataset.
            with zipfile.ZipFile(os.path.join(self.dataset_dir, '{}.zip'.format(self.dataset)), "r") as zip_ref:
                zip_ref.extractall(self.dataset_dir)

        if not os.path.exists(os.path.join(self.config['PATHS']['annotations'], '{}_video_level_annotation.csv'.format(self.dataset))):
            # Download dataset classes file.
            gdown.download(id=csv_file_id,
                           output=os.path.join(self.config['PATHS']['annotations'], '{}_video_level_annotation.csv'.format(self.dataset)))

    def summary(self):
        print('\nDataset: {}'.format(self.dataset))
        print('='*30)
        print('Paper: {}\nRelease year: {}\nResource: {}\nHours length: {}\nNumber of instances: {}'.format(self.paper, self.release_year, self.resource, self.hours_length, self.number_of_instances))
        print('=' * 30)
        print('Classes distribution:')
        for key in self.classes_distribution:
            print('• {} ({})'.format(key, self.classes_distribution[key]))


    def new_split(self, props, names):
        """Create a JSON file with dataset splitting.

        :param props: List with the proportions used to split.
        :type props:  :class:`list`
        :param names: List with the partition names.
        :type names: :class:`list`
        :returns: :class:`str` -- split identifier hash.
        """

        # Exception handling.
        if len(props) != len(names):
            raise Exception("The 'props' and 'names' parameters must have the same sizes")
        elif not all(isinstance(x, (float, int)) for x in props):
            raise Exception("All values in 'props' parameter must be of type float or int")
        elif max(props) > 1 or min(props) < 0 or sum(props) != 1.0:
            raise Exception("'props' values must be between 0 and 1 and sum equal to 1")
        elif not all(isinstance(x, str) for x in names):
            raise Exception("All values in 'names' parameter must be of type str")

        name_videos = os.listdir(os.path.join(self.dataset_dir, self.dataset))
        # Save partition names as dictionary keys.
        partitions = dict.fromkeys(names, pd.DataFrame())
        # Separate according to list of proportions and save in dictionary with partitions.
        splited = self.__split_by_props(pd.DataFrame(name_videos), props)
        for i, key in enumerate(partitions):
            partitions[key] = pd.concat([partitions[key], splited[i]]).squeeze()
        # Convert the Series to list.
        for key in partitions:
            partitions[key] = partitions[key].tolist()
        # Reference to the dataset name.
        partitions['dataset'] = self.dataset

        # Save the split to a JSON file with an identifying hash.
        hash = str(random.getrandbits(128))
        with open(os.path.join(self.splits_dir, 'split_{}.json'.format(hash)), "w") as out_file:
            json.dump(partitions, out_file, indent=2)

        return hash

    def load_split(self, hash):
        """Read a JSON file with dataset splitting.

        :param hash: split identifier hash.
        :type hash:  :class:`str`
        :returns: :class:`dict` -- dictionary with dataset split.
        """

        with open(os.path.join(self.splits_dir, 'split_{}.json'.format(hash))) as json_file:
            return json.load(json_file)

    def load_data_gen(self, split, partition, category, annotation, classes, modes, target_size, batch_size, crop=False, samples_per_class=None, sequence_size=None, intra_sequence_skip=None, inter_sequence_skip=1):
        """Create a Custom Data Generator from a split.

        :param split: Dictionary with dataset split.
        :type split:  :class:`dict`
        :param partition: Identifier key for split partition.
        :type category:  :class:`str`
        :param category: Reference to training category.
        :type partition:  :class:`str`
        :param annotation: Reference to annotation file.
        :type annotation:  :class:`str`
        :param classes: List with classes that will be used.
        :type classes:  :class:`list`
        :param modes: List of frame representation modes used for extraction.
        :type modes:  :class:`list`
        :param target_size: Dimension used to resize the frame.
        :type target_size:  :class:`tuple`
        :param batch_size: Size of the batches of data.
        :type batch_size:  :class:`int`
        :param samples_per_class: Limit of examples of a class in each split.
        :type samples_per_class:  :class:`int`

        :param sequence_size: Frame sequence size for sequential Sequential Classification.
        :type sequence_size:  :class:`int`
        :param intra_sequence_skip: Intra-interval of frames for a sample in Sequential Classification.
        :type intra_sequence_skip:  :class:`int`
        :param inter_sequence_skip: Inter-interval for frame extraction.
        :type inter_sequence_skip:  :class:`int`
        :returns: :class:`DataGen` -- Custom object to train TensorFlow models.
        """

        # Exception handling.
        if partition not in split:
            raise Exception("The '{}' partition does not exist.".format(partition))
        for mode in modes:
            if mode not in ['frame_rgb', 'frame_gray', 'flow']:
                raise Exception("'{}' is not a mode option.".format(mode))

        annotation_file = pd.read_csv(os.path.join(self.config['PATHS']['annotations'], '{}_{}.csv'.format(self.dataset, annotation)))
        # Selects only the DataFrame rows with the class passed by parameter.
        annotation_file = annotation_file[annotation_file['label'].isin(classes)]
        n_label = annotation_file['label'].nunique()
        id_label = annotation_file['label'].unique()

        extracted_frames = pd.DataFrame(columns=['name', 'path', 'modes', 'index', 'label'])
        # List with possible videos existing in the split.
        candidate_videos = annotation_file['name'].tolist()

        for video in split[partition]:
            if video in candidate_videos:
                if annotation == 'video_level_annotation':
                    video_path = os.path.join(self.dataset_dir, self.dataset, video)
                    frames = self.__count_frames(video_path)
                    if category == 'framewise_classification':
                        for i in range(0, frames - 1, inter_sequence_skip):
                            label = annotation_file[annotation_file['name'] == video]['label'].item()
                            new_row = pd.DataFrame([{'name': video, 'path': video_path, 'index': i, 'label': label}])
                            extracted_frames = pd.concat([extracted_frames, new_row])
                    elif category == 'sequential_classification':
                        for i in range(0, frames-(intra_sequence_skip*(sequence_size-1)), (intra_sequence_skip*(sequence_size-1))+inter_sequence_skip):
                            label = annotation_file[annotation_file['name'] == video]['label'].item()
                            new_row = pd.DataFrame([{'name': video, 'path': video_path, 'index': i, 'label': label}])
                            extracted_frames = pd.concat([extracted_frames, new_row])
                else:
                    tmp = annotation_file[annotation_file.name == video]
                    tmp['path'] = os.path.join(self.dataset_dir, self.dataset, video)
                    extracted_frames = pd.concat([extracted_frames, tmp])

        # Fill column 'modes'.
        extracted_frames['modes'] = str(modes)
        # Limit samples per class.
        if samples_per_class != None:
            extracted_frames = extracted_frames.groupby('label').apply(lambda x: x.sample(n=samples_per_class)).reset_index(drop=True)
            extracted_frames = extracted_frames.sample(frac=1).reset_index(drop=True)

        if category == 'framewise_classification':
            # Framewise DataGen.
            datagen = DataGenFramewiseClassification(extracted_frames,
                                    batch_size=batch_size,
                                    input_size=target_size,
                                    id_label=id_label,
                                    n_label=n_label,
                                    crop=crop)
        elif category == 'sequential_classification':
            # If an external annotation file exists, update the 'intra_sequence_skip' and 'sequence_size' values.
            if annotation != 'video_level_annotation':
                sequence_size = int(extracted_frames.iloc[0]['size'])
                intra_sequence_skip = int(extracted_frames.iloc[0]['intra'])
            # Sequential DataGen.
            datagen = DataGenSequentialClassification(extracted_frames,
                                    batch_size=batch_size,
                                    input_size=target_size,
                                    id_label=id_label,
                                    n_label=n_label,
                                    sequence_size=sequence_size,
                                    intra_sequence_skip=intra_sequence_skip,
                                    crop=crop)

        return datagen
