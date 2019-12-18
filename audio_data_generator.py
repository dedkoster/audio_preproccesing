"""Utilities for real-time audio data augmentation.
URL:
- https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py
"""

import os
import threading
import numpy as np
import pandas as pd
from glob import glob1 as glob
from librosa.output import write_wav
from librosa.effects import time_stretch
from librosa.core import load as load_audio

from keras_preprocessing import get_keras_submodule

backend = get_keras_submodule('backend')
keras_utils = get_keras_submodule('utils')


class AudioDataGenerator(object):
    """Generate batches of audio data with real-time data augmentation.
         The data will be looped over (in batches).

        # Arguments
            shift: Float > 0.
                Range of random shift. Randomly adds a shift at the beginning or end of the audio data.
            noise: Float > 0 (default: 0.005).
                Volume of added white noise.
            volume: Float > 0.
                Volume of audio data.
            stretch: Float > 0.
                Stretch factor.
                If `rate > 1`, then the signal is sped up.
                If `rate < 1`, then the signal is slowed down.
            bg_noise_dir: String.
                The path to the directory with audio files of background noise.
            bg_noise_volume: Float (default: 0.1).
                (only relevant if `bg_noise_dir` is set).
                Volume of background noise.

        # Examples
            Example of using `.flow(x, y)`:

            ```python
            df_data = pd.read_excel('data.xlsx')
            datagen = AudioDataGenerator(noise=0, bg_noise_dir='background_noise')

            # fits the model on batches with real-time data augmentation:
             model.fit_generator(datagen.flow(dataframe_data=df_data, batch_size=2),
                        steps_per_epoch=len(x_train) / 2, epochs=epochs)

            # sortagrad
            for e in range(epochs):
                print('Epoch', e)
                if epochs < 2:
                    sortagrad = 10
                elif epochs < 6:
                    sortagrad = 20
                else:
                    sortagrad = None

                for x_batch, y_batch in datagen.flow(dataframe_data=df_data, batch_size=batch_size, sorting='asc',
                                                     top_count=sortagrad):
                    model.fit(x_batch, y_batch)
            ```
    """

    def __init__(self,
                 shift=0.,
                 noise=0.005,
                 volume=1.,
                 stretch=0.,
                 bg_noise_dir='',
                 bg_noise_volume=.05):

        self.shift = shift
        self.noise = noise
        self.stretch = stretch
        self.volume = volume
        self.bg_noise_dir = bg_noise_dir
        self.bg_noise_volume = bg_noise_volume

        if self.shift:
            if self.shift < 0.:
                raise ValueError("Shift can't be negative number")

        if self.noise:
            if self.noise < 0.:
                raise ValueError("Noise can't be negative number")

        if self.volume:
            if self.volume < 0.:
                raise ValueError("Volume can't be negative number")

        if self.stretch:
            if self.stretch < 0.:
                raise ValueError("Volume can't be negative number")

    def flow(self, dataframe_data=None, batch_size=32, sorting=None,
             save_to_dir=None,
             save_prefix='', stuffing=0., top_count=None, seed=None, shuffle=None):
        """Takes data & label from dataframe, generates batches of augmented data.

            # Arguments
                dataframe_data: DataFrame.
                    The dataframe of audio data.
                    It must contain 3 columns: transcript, wav_filename, wav_filesize.
                batch_size: Int (default: 32).
                sorting: None or string.
                    Sorts audio data by size in ascending (asc) or descending (desc) order.
                save_to_dir: None or str (default: None).
                    This allows you to optionally specify a directory
                    to which to save the augmented audio data being generated
                    (useful for listening what you are doing).
                save_prefix: Str (default: `''`).
                    Prefix to use for filenames of saved audio data
                    (only relevant if `save_to_dir` is set).
                stuffing: Float (default: 0.).
                    Range from -1 to 1.
                    The value that fills the audio when the size increases.
                top_count: None or int.
                    Number of first audio from the dataset used for fitting.
                seed: Int (default: None).

            # Returns
                An `DataFrameIterator` yielding tuples of `(x, y)`
                    where `x` is a numpy array of augmented audio data
                    and `y` is a numpy array of corresponding labels.
        """
        return DataFrameIterator(self, dataframe_data=dataframe_data,
                                 batch_size=batch_size, sorting=sorting,
                                 save_to_dir=save_to_dir,
                                 save_prefix=save_prefix,
                                 stuffing=stuffing, top_count=top_count, shuffle=shuffle, seed=seed)

    def transform(self, x, sr):
        """Applies the audio conversion.

            # Arguments
                x: numpy array.
                    The audio signal.
                sr: int.
                    Audio sample rate.

            # Returns
                A transformed version (x, sr) of the input.
        """
        input_length = len(x)

        if self.stretch:
            x = time_stretch(x, self.stretch)
            if len(x) > input_length:
                x = x[:input_length]
            else:
                x = np.pad(x, (0, max(0, int(input_length - len(x)))), "constant")

        if self.shift:
            x = np.pad(x, (int(self.shift * sr), 0) if np.random.random() < 0.5 else (0, int(self.shift * sr)),
                       'constant')
            if len(x) > input_length:
                x = x[:input_length]

        if self.noise:
            x = x + self.noise * np.random.randn(len(x))

        if self.bg_noise_dir:
            bg_noise_data = glob(self.bg_noise_dir, "*.wav")
            index_chosen_bg_file = int(len(bg_noise_data) * np.random.random())
            x_bg, sr_bg = load_audio(os.path.join(self.bg_noise_dir, bg_noise_data[index_chosen_bg_file]))
            x_bg_rand = x_bg[int(len(x_bg) * np.random.random()):]
            while input_length > len(x_bg_rand):
                x_bg_rand = np.concatenate([x_bg_rand, x_bg])

            if len(x_bg_rand) > input_length:
                x_bg = x_bg_rand[:input_length]

            x = x + (x_bg * self.bg_noise_volume)

        if self.volume:
            x = x * self.volume

        return x, sr


class Iterator(keras_utils.Sequence):
    """Base class for audio data iterators.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.epochs = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self.epochs += 1
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class DataFrameIterator(Iterator):
    """Iterator capable of reading audio data from the dataframe.

        # Arguments
            audio_data_generator: Instance of `AudioDataGenerator`
                to use for transformations audio data.
            dataframe_data: DataFrame.
                The dataframe of audio data.
                It must contain 3 columns: transcript, wav_filename, wav_filesize.
            batch_size: Int (default: 32).
                Size of a batch/
            sorting: None or string.
                Sorts audio data by size in ascending (asc) or descending (desc) order.
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented audio data being generated
                (useful for listening what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved audio data
                (only relevant if `save_to_dir` is set).
            stuffing: Float (default: 0.).
                Range from -1 to 1.
                The value that fills the audio when the size increases.
            top_count: None or int.
                Number of first audio from the dataset used for fitting.
            seed: Int (default: None).
    """

    def __init__(self, audio_data_generator, dataframe_data=None,
                 batch_size=32, sorting=None,
                 save_to_dir=None, save_prefix='', stuffing=0., top_count=None, shuffle=None, seed=None):
        self.dataframe_data = dataframe_data
        self.sorting = sorting
        self.top_count = top_count
        self.audio_data_generator = audio_data_generator
        self.save_prefix = save_prefix
        self.save_to_dir = save_to_dir
        self.stuffing = stuffing
        self.shuffle = shuffle

        if isinstance(self.dataframe_data, pd.DataFrame):
            if self.dataframe_data.columns.size != 3:
                raise ValueError("Your dataframe must have this columns: transcript, wav_filename, wav_filesize")

            self.dataframe_data.columns = np.arange(len(self.dataframe_data.columns))
        else:
            raise ValueError("Pass the dataframe in this parameter.")

        if 1. < stuffing < -1.:
            raise ValueError("Stuffing must be in the range from -1 to 1.")

        if self.sorting:
            if self.sorting not in ('asc', 'desc'):
                raise ValueError("sorting can be: 'asc' or 'desc'")

            if self.sorting == 'asc':
                ascend = True
            elif self.sorting == 'desc':
                ascend = False

            self.dataframe_data.sort_values([2], ascending=ascend, inplace=True)
            if self.top_count:
                self.dataframe_data = self.dataframe_data.head(self.top_count)

        self.samples = len(self.dataframe_data)

        if self.sorting:
            self.dataframe_data.index = range(self.samples)

        if self.save_to_dir:
            if not os.path.exists(self.save_to_dir):
                os.makedirs(self.save_to_dir)

        super(DataFrameIterator, self).__init__(self.samples,
                                                batch_size,
                                                self.shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        print("Batch index:", self.batch_index)

        index_array.sort()
        # find max size in batch
        filtered_df = self.dataframe_data.loc[self.dataframe_data.index.isin(index_array)]
        bigfile_in_batch = filtered_df.loc[filtered_df[2].idxmax()]
        max_audiosize_in_batch = int(bigfile_in_batch[2])

        # when stretching slow down the audio we change max_audiosize_in_batch by stretch rate
        if self.audio_data_generator.stretch and (self.audio_data_generator.stretch < 1):
            max_audiosize_in_batch = int(max_audiosize_in_batch * (1 + self.audio_data_generator.stretch))

        # when shift is happens we change max_audiosize_in_batch accordingly
        if self.audio_data_generator.shift:
            _, max_sr = load_audio(bigfile_in_batch[1])
            max_audiosize_in_batch = int(max_audiosize_in_batch + (self.audio_data_generator.shift * max_sr))

        batch_x = np.zeros(
            (len(index_array),) + (max_audiosize_in_batch,),
            dtype=backend.floatx())
        batch_y = [0] * len(index_array)

        for i, j in enumerate(index_array):
            current_audiofile = self.dataframe_data.iloc[j]
            y = current_audiofile[0]
            x, sr = load_audio(current_audiofile[1])

            if len(x) < max_audiosize_in_batch:
                x = np.pad(x, (0, max(0, int(max_audiosize_in_batch - len(x)))), "constant",
                           constant_values=(self.stuffing))

            x, sr = self.audio_data_generator.transform(x, sr)

            # optionally save augmented audio to disk for debugging purposes
            if self.save_to_dir:
                fname = '{prefix}_{index}_{hash}.wav'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e7))
                write_wav(os.path.join(self.save_to_dir, fname), x, sr)

            batch_x[i] = x
            batch_y[i] = y

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)
