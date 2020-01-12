from DatasetFlow import DatasetFlow
import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder
import os
from timer import Timer
import pickle


def save_pkl(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Reading data...')
dataset_flow = DatasetFlow()
dataset_flow._init_from_file('data_pairs2.py')
dataset_flow.tokenize_code()
save_pkl(dataset_flow, 'datasetflow2.pkl')

print('Defining and creating model...')
BUFFER_SIZE = len(dataset_flow.input_tensor)
BATCH_SIZE = 13
embedding_dim = 128
units = 256
vocab_inp_size = len(dataset_flow.input_tokenizer.word_index) + 1
vocab_target_size = len(dataset_flow.target_tokenizer.word_index) + 1
print('vocabularies: input [{}] | target [{}]'.format(vocab_inp_size, vocab_target_size))
max_len_input, max_len_target = dataset_flow.max_sequence_length()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_inp_size, embedding_dim, mask_zero=True, input_length=max_len_input))
model.add(tf.keras.layers.LSTM(units))
model.add(tf.keras.layers.RepeatVector(max_len_target))
model.add(tf.keras.layers.LSTM(units, return_sequences=True))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_target_size, activation='softmax')))

model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
print(model.summary())

# model = tf.keras.models.load_model('model_checkpoint/lstm_model2.h5')
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# monitor, mode = 'acc', 'max'
monitor, mode = 'loss', 'min'

model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoint/lstm_model4.h5', monitor=monitor, mode=mode, verbose=1, save_best_only=True, save_weights_only=False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, min_lr=1e-5, verbose=1, mode=mode)
model.fit(dataset_flow.input_tensor,
          dataset_flow.target_tensor,
          epochs=200,
          batch_size=BATCH_SIZE,
          callbacks=[model_checkpoint, reduce_lr],
          )
