import tokenize
import tensorflow as tf
import numpy as np


def get_tokens_from_file(file_path):
    all_tokens = []
    with open(file_path, 'rb') as file:
        for five_tuple in tokenize.tokenize(file.readline):
            token = five_tuple.string.strip()
            if token == '' or '#' in token or token == 'utf-8':
                continue
            else:
                all_tokens.append(token)
    return all_tokens


def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = tf.keras.utils.to_categorical(sequence, num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


def seq_to_words(sequence, tokenizer):
    words = []
    for id_ in sequence:
        word = tokenizer.index_word[id_]
        words.append(word)
        if word == '<end>':
            break
    return words


class DatasetFlow:
    def __init__(self):
        self.target_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.input_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.input_sequences = []
        self.target_sequences = []
        self.input_sequences_test = []
        self.target_sequences_test = []
        self.dictionary = {'<start>', '<end>'}
        self.test_words_translations = []
        self.test_words_translations_inverse = []
        self.index_vars = ['i', 'j', 'k']
        self.quantity_vars = ['n', 'm', '1']
        self.array_names = ['a', 'b', 'c']

    def translate_tokens(self, tokens):
        '''
        gets raw tokens and transforms them to own code
        '''
        translation_dict = {}
        translation_dict_inverse = {}
        for i, token in enumerate(tokens):
            unknown_tokens = []
            replacement_lists = []
            if token == 'for':
                unknown_tokens.append(tokens[i+1])
                replacement_lists.append(self.index_vars)
            elif token == 'range':
                unknown_tokens.append(tokens[i + 2])
                replacement_lists.append(self.quantity_vars)
            elif token == '[':
                # unknown index var
                unknown_tokens.append(tokens[i + 1])
                replacement_lists.append(self.index_vars)

                # unknown array var
                if tokens[i-1] != ']':
                    unknown_tokens.append(tokens[i-1])
                    replacement_lists.append(self.array_names)
            elif (token == '=' or token == '+=') and tokens[i-1] != ']':
                unknown_tokens.append(tokens[i-1])
                replacement_lists.append(self.array_names)
            else:
                continue

            for unknown_token, replacement_list in zip(unknown_tokens, replacement_lists):
                if unknown_token not in translation_dict:
                    unique_replacement = [element for element in replacement_list if element not in translation_dict_inverse][0]
                    translation_dict[unknown_token] = unique_replacement
                    translation_dict_inverse[unique_replacement] = unknown_token
        print('Before: ', tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            if token in translation_dict:
                tokens[i] = translation_dict[token]
        print('After:', tokens, '\n')
        self.test_words_translations.append(translation_dict)
        self.test_words_translations_inverse.append(translation_dict_inverse)
        return tokens

    def translate_tokens_target(self, tokens):
        print('Before:', tokens)
        for i in range(len(tokens)):
            token = tokens[i]
            for dict_ in self.test_words_translations:
                if token in dict_:
                    tokens[i] = dict_[token]
        print('After:', tokens, '\n')
        return tokens

    def _init_from_file_no_target(self, file_path):
        tokens = get_tokens_from_file(file_path)
        input_sequences = self.input_sequences_test

        sequence = ['<start>']
        for token in tokens:
            if token == 'next':
                sequence.append('<end>')
                input_sequences.append(sequence)
                sequence = ['<start>']
            else:
                self.dictionary.add(token)
                sequence.append(token)
        sequence.append('<end>')
        input_sequences.append(sequence)

        self.input_sequences_test = [self.translate_tokens(tokens) for tokens in self.input_sequences_test]

    def _init_from_file(self, file_path, for_test=False):
        tokens = get_tokens_from_file(file_path)
        input_sequences = self.input_sequences
        target_sequences = self.target_sequences

        if for_test:
            input_sequences = self.input_sequences_test
            target_sequences = self.target_sequences_test

        sequence = []
        for token in tokens:
            if token == 'next':
                if len(sequence) > 0:
                    sequence.append('<end>')
                    target_sequences.append(sequence)
                sequence = ['<start>']
            elif token == 'translation':
                sequence.append('<end>')
                input_sequences.append(sequence)
                sequence = ['<start>']
            else:
                self.dictionary.add(token)
                sequence.append(token)
        sequence.append('<end>')
        target_sequences.append(sequence)

        if for_test:
            self.input_sequences_test = [self.translate_tokens(tokens) for tokens in self.input_sequences_test]
            self.target_sequences_test = [self.translate_tokens_target(tokens) for tokens in self.target_sequences_test]

    def init_from_file(self, file_path):
        with open(file_path, 'rb') as file:
            sequence = []
            for five_tuple in tokenize.tokenize(file.readline):
                token = five_tuple.string.strip()
                if token == '' or '#' in token or token == 'utf-8':
                    continue
                elif token == 'next':
                    if len(sequence) > 0:
                        sequence.append('<end>')
                        self.target_sequences.append(sequence)
                    sequence = ['<start>']
                elif token == 'translation':
                    sequence.append('<end>')
                    self.input_sequences.append(sequence)
                    sequence = ['<start>']
                else:
                    self.dictionary.add(token)
                    sequence.append(token)
            sequence.append('<end>')
            self.target_sequences.append(sequence)

            self.input_sequences = [' '.join(elements) for elements in self.input_sequences]
            self.target_sequences = [' '.join(elements) for elements in self.target_sequences]

        print('number of pairs: ', len(self.input_sequences), self.input_sequences[0])

    def tokenize_code_test(self):
        self.input_tensor_test = self.input_tokenizer.texts_to_sequences(self.input_sequences_test)
        self.input_tensor_test = tf.keras.preprocessing.sequence.pad_sequences(self.input_tensor_test, padding='post')

        self.target_tensor_test = self.target_tokenizer.texts_to_sequences(self.target_sequences_test)
        self.target_tensor_test = tf.keras.preprocessing.sequence.pad_sequences(self.target_tensor_test, padding='post')
        self.target_tensor_test = encode_output(self.target_tensor_test, len(self.target_tokenizer.word_index) + 1)

    def tokenize_code_test_no_target(self):
        self.input_tensor_test = self.input_tokenizer.texts_to_sequences(self.input_sequences_test)
        self.input_tensor_test = tf.keras.preprocessing.sequence.pad_sequences(self.input_tensor_test, padding='post', maxlen=36)
        print()
        # self.target_tensor_test = self.target_tokenizer.texts_to_sequences(self.target_sequences_test)
        # self.target_tensor_test = tf.keras.preprocessing.sequence.pad_sequences(self.target_tensor_test, padding='post')
        # self.target_tensor_test = encode_output(self.target_tensor_test, len(self.target_tokenizer.word_index) + 1)

    def tokenize_code(self):
        self.input_tokenizer.fit_on_texts(self.input_sequences)
        self.input_tensor = self.input_tokenizer.texts_to_sequences(self.input_sequences)
        self.input_tensor = tf.keras.preprocessing.sequence.pad_sequences(self.input_tensor, padding='post')

        self.target_tokenizer.fit_on_texts(self.target_sequences)
        self.target_tensor = self.target_tokenizer.texts_to_sequences(self.target_sequences)
        self.target_tensor = tf.keras.preprocessing.sequence.pad_sequences(self.target_tensor, padding='post')
        self.target_tensor = encode_output(self.target_tensor, len(self.target_tokenizer.word_index) + 1)

        print('Input shape:', self.input_tensor.shape)
        print('Target shape:', self.target_tensor.shape)

    def max_sequence_length(self):
        max_input = max(len(tensor) for tensor in self.input_tensor)
        max_target = max(len(tensor) for tensor in self.target_tensor)
        print('Max input {}, max target {}'.format(max_input, max_target))
        return max_input, max_target

    # one hot encode target sequence

    def input_seq_to_words(self, sequence):
        return seq_to_words(sequence, self.input_tokenizer)

    def target_seq_to_words(self, sequence):
        prediction_ids = [np.argmax(output) for output in sequence]
        return seq_to_words(prediction_ids, self.target_tokenizer)

    def translate_back(self, sequences, tokenizer):
        translated_sequences = []

        for sequence, translation_dict_inverse in zip(sequences, self.test_words_translations_inverse):
            words = seq_to_words(sequence, tokenizer)
            for i, word in enumerate(words):
                if word in translation_dict_inverse:
                    words[i] = translation_dict_inverse[word]
            translated_sequences.append(words)

        return translated_sequences
