import os
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from librosa.core import load as load_audio
from audio_data_generator import AudioDataGenerator

epochs = 5
batch_size = 2

a = Input(shape=(batch_size,))
b = Dense(batch_size)(a)
model = Model(inputs=a, outputs=b)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

df_data = pd.read_excel('data.xlsx')


# this function add filesize to dataframe and save as .xlsx
def add_audiosize_to_dataframe(dataframe):
    dataframe["wav_filesize"] = None
    for index, row in dataframe.iterrows():
        try:
            x_bg, sr_bg = load_audio(os.path.join("records_sample", row["wav_filename"] + ".wav"))
            if x_bg.size < 100:
                dataframe.drop(index)
                continue
        except:
            dataframe.drop(index)
            continue
        dataframe.at[index, 'wav_filename'] = os.path.join("records_sample", row["wav_filename"] + ".wav")
        dataframe.at[index, 'wav_filesize'] = x_bg.size

    return dataframe.to_excel("data.xlsx")


# add_audiosize_to_dataframe(data)

datagen = AudioDataGenerator(noise=0, bg_noise_dir='background_noise', stretch=0.6, shift=1.)

# for test
train_data = datagen.flow(dataframe_data=df_data, batch_size=batch_size, sorting='asc', save_to_dir="test")
for i in range(2):
    train_data.next()

# for sortagrad
# for e in range(epochs):
#     print('Epoch', e)
#     if epochs < 2:
#         sortagrad = 10
#     elif epochs < 6:
#         sortagrad = 20
#     else:
#         sortagrad = None
#
#     for x_batch, y_batch in datagen.flow(dataframe_data=df_data, batch_size=batch_size, sorting='asc',
#                                          top_count=sortagrad):
#         model.fit(x_batch, y_batch)
