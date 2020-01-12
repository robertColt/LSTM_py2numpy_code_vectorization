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
dataset_flow.init_from_file('data_pairs.py')
dataset_flow.tokenize_code()
save_pkl(dataset_flow, 'datasetflow.pkl')

print('Defining and creating model...')
BUFFER_SIZE = len(dataset_flow.input_tensor)
BATCH_SIZE = 2
embedding_dim = 128
units = 512
vocab_inp_size = len(dataset_flow.input_tokenizer.word_index) + 1
vocab_target_size = len(dataset_flow.target_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((dataset_flow.input_tensor, dataset_flow.target_tensor)).shuffle(
    BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
print('Example batches input/target:', example_input_batch.shape, example_target_batch.shape)

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


@tf.function
def train_step(inp, target, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([dataset_flow.target_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(1, target.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(target[:, t], predictions)

            dec_input = tf.expand_dims(target[:, t], 1)
        batch_loss = (loss / int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


print('Starting training...')
EPOCHS = 100
steps_per_epoch = len(dataset_flow.input_tensor) // BATCH_SIZE
timer = Timer()
previous_loss = 1e5

for epoch in range(EPOCHS):
    timer.start()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    print('Something', end='\r')
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        print("Epoch {} batch [{}/{}]".format(epoch + 1, batch + 1, steps_per_epoch), end='\r', flush=True)
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    total_loss = total_loss / steps_per_epoch

    if total_loss < previous_loss:
        print_str = 'Loss improved from {:.4f} to {:.4f} Saving model to file...'.format(previous_loss, total_loss)
        print(print_str)
        encoder.save_weights('model_checkpoint/encoder.h5')
        decoder.save_weights('model_checkpoint/decoder.h5')
    previous_loss = total_loss

    print('Epoch {} Loss {:.4f} : {:.0f} s'.format(epoch + 1,
                                                   total_loss,
                                                   timer.stop()))
