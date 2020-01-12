import tensorflow as tf
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from DatasetFlow import DatasetFlow
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def pretty_print(words):
    output_str = ''
    for_cnt = 0
    for i, word in enumerate(words):
        if word == '<start>':
            continue
            output_str += word + '\n'
        elif word == '<end>':
            continue
            output_str += '\n' + word
        elif word == 'for':
            for_cnt += 1
            output_str += word + ' '
        elif word == 'in':
            output_str += ' ' + word + ' '
        elif word == ':':
            output_str += ':\n' + ''.join('\t' for i in range(for_cnt))
        elif word == ']' and words[i+1] not in '[ += = * ,'.split(' '):
            output_str += word + '\n'
        elif word in ', + += = * /'.split(' '):
            output_str += ' ' + word + ' '
        else:
            output_str += word

    return output_str


if __name__ == '__main__':
    model_path = 'model_checkpoint/lstm_model4.h5'
    model = tf.keras.models.load_model(model_path)

    with open('datasetflow2.pkl', 'rb') as dataset_file:
        dataset_flow: DatasetFlow = pickle.load(dataset_file)

    # pair_id = -3
    # inputs = dataset_flow.input_tensor[pair_id]
    # target = dataset_flow.target_tensor[pair_id]
    # print(inputs.shape, target.shape)
    #
    # prediction = model.predict(inputs.reshape((1, 49,)), verbose=0)[0]
    #
    # print('Input:', dataset_flow.input_seq_to_words(inputs))
    # print('Target', dataset_flow.target_seq_to_words(target))
    # print('Predicted', dataset_flow.target_seq_to_words(prediction))

    for_test = True
    dataset_flow._init_from_file('data_pairs_test2.py', for_test=for_test)
    dataset_flow.tokenize_code_test()

    predictions = model.predict(dataset_flow.input_tensor_test)

    prediction_ids = np.argmax(predictions, axis=2)
    expected_ids = np.argmax(dataset_flow.target_tensor_test, axis=2)

    input_words = dataset_flow.translate_back(dataset_flow.input_tensor_test, dataset_flow.input_tokenizer)
    expected_words = dataset_flow.translate_back(expected_ids, dataset_flow.target_tokenizer)
    prediction_words = dataset_flow.translate_back(prediction_ids, dataset_flow.target_tokenizer)

    for inp, expected, prediction in zip(input_words, expected_words, prediction_words):
        print('INPUT:\n', pretty_print(inp))
        print('EXPECTED:', pretty_print(expected))
        print('PREDICTED:', pretty_print(prediction))
        print()


