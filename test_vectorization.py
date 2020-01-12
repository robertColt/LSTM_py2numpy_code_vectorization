import tensorflow as tf
import pickle
from Encoder import Encoder
from Decoder import Decoder


def evaluate(encoder, decoder, sentence, inp_lang_tokenizer, target_lang_tokenizer, hidden_units, max_len_input,
             max_len_target):
    inputs = [inp_lang_tokenizer.word_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], max_len=max_len_input, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''

    hidden = [tf.zeros((1, hidden_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_len_target):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        target_word = target_lang_tokenizer.index_word[predicted_id]
        result += target_word + ' '

        if target_word == '<end>':
            return result, sentence

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence


if __name__ == '__main__':
    with open('datasetflow.pkl', 'rb') as dataset_file:
        dataset_flow = pickle.load(dataset_file)

    BATCH_SIZE = 2
    vocab_inp_size = len(dataset_flow.input_tokenizer.word_index) + 1
    vocab_target_size = len(dataset_flow.target_tokenizer.word_index) + 1
    embedding_dim = 128
    units = 512
    print(vocab_inp_size, vocab_target_size)
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_target_size, embedding_dim, units, BATCH_SIZE)
    encoder.initialize_hidden_state()
    encoder.load_weights('model_checkpoint/encoder.h5')
    decoder.load_weights('model_checkpoint/decoder.h5')

