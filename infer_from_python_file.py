import pickle
from typing import List
import re
import tensorflow as tf
import numpy as np

from DatasetFlow import DatasetFlow
from test_vectorization2 import pretty_print


def should_ignore_line(line: str):
    if line.startswith('from') and 'import' in line \
            or line.startswith('import') \
            or line == '\n'\
            or line.strip().startswith('#'):
        return True
    else:
        return False


def indent_number(line):
    space_cnt = 0
    for char in line:
        if char == ' ':
            space_cnt += 1
        else:
            return space_cnt // 2

def is_already_found(line_number, start_end_list):
    for s, e in start_end_list:
        if s == line_number:
            return True
    return False

class PyFileProcessor:
    def __init__(self, python_file_path):
        self.code_lines = []
        self.inside_multi_comment = False
        self.multi_comment_end = False
        self.process_file(python_file_path)

    def process_file(self, python_file_path):
        with open(python_file_path, 'rt') as python_file:
            self.code_lines = python_file.readlines()

    def ignore_multi_comment(self, line, verbose=False):
        '''
        returns True if the line should be ignored
        '''
        stripped_line: str = line.strip()
        if stripped_line.startswith('\"\"\"') or \
                stripped_line.startswith('\'\'\'') or \
                stripped_line.startswith('r\"\"\"'):
            if self.inside_multi_comment:
                if verbose:
                    print('>>>Finish multi comment')
                self.inside_multi_comment = False
                return True
            else:
                if verbose:
                    print('>>>Start multi comment')
                self.inside_multi_comment = True
                return True
        return self.inside_multi_comment

    def test_code_comprehension(self):
        for i, line in enumerate(self.code_lines[:200]):
            ignore_comment = self.ignore_multi_comment(line, verbose=True)
            line_to_print = '{} {} == {} ... {} {}'.format(i + 1,
                                                        repr(line),
                                                        should_ignore_line(line),
                                                        indent_number(line),
                                                        'C' if ignore_comment else '')
            print(line_to_print)

    def detect_dot_product(self):
        prev_line_indent = 0
        for_started = False
        dot_product_line_spans = []
        start_line, end_line = 0, 0
        index_name = ''
        for line_number, line in enumerate(self.code_lines):
            if should_ignore_line(line) or self.ignore_multi_comment(line):
                continue
            current_line_indent = indent_number(line)
            stripped_line = line.strip()
            if for_started:
                if current_line_indent == prev_line_indent+1:
                    line_elements = re.split('\[|\]| |', stripped_line)
                    line_elements = [x for x in line_elements if x != '']
                    if len(line_elements) == 8 \
                            and line_elements[1] == index_name \
                            and line_elements[4] == index_name \
                            and line_elements[7] == index_name \
                            and line_elements[2] == '=' \
                            and line_elements[5] == '*':
                        end_line = line_number
                        dot_product_line_spans.append((start_line, end_line))
                        for_started = False
                elif current_line_indent == prev_line_indent:
                    for_started = False
            else:
                if stripped_line.startswith('for'):
                    for_elements: List[str] = stripped_line.split(' ')
                    last_element = for_elements[-1]  # range(...):
                    if len(for_elements) != 4\
                            or (not (last_element.startswith('range(') and last_element.endswith('):'))):
                        continue
                    index_name = for_elements[1]
                    for_started = True
                    prev_line_indent = current_line_indent
                    start_line = line_number
                    # accumulated_tokens.extend(for_elements[:3])
                    # accumulated_tokens.append('(')
                    # last_element_tokens = re.split('(|)', last_element) # should be [range, inside_range, :]
                    # accumulated_tokens.append()
        return dot_product_line_spans

    def detect_average(self, already_found_line_spans):
        prev_line_indent = 0
        for_started = False
        average_detected = False
        avg_line_spans = []
        start_line, end_line, avg_line = 0, 0, 0
        index_name = ''
        accumulator = ''
        num_elems = ''
        for line_number, line in enumerate(self.code_lines):
            if should_ignore_line(line) or self.ignore_multi_comment(line):
                continue
            current_line_indent = indent_number(line)
            stripped_line = line.strip()
            if for_started:
                line_elements = re.split('\[|\]| |', stripped_line)
                line_elements = [x for x in line_elements if x != '']
                if current_line_indent == prev_line_indent+1:
                    if len(line_elements) == 4 \
                            and line_elements[-1] == index_name \
                            and line_elements[1] == '+=':
                        accumulator = line_elements[0]
                        average_detected = True
                        avg_line = line_number
                elif current_line_indent == prev_line_indent:
                    # need to find average line
                    if len(line_elements) == 5 \
                            and line_elements[0] == accumulator \
                            and line_elements[1] == '=' \
                            and line_elements[2] == accumulator \
                            and line_elements[3] == '/':

                        end_line = line_number
                        avg_line_spans.append((start_line, avg_line, end_line))
                        for_started = False
                elif current_line_indent == prev_line_indent - 1:
                    for_started = False
            else:
                if is_already_found(line_number, already_found_line_spans): continue
                if stripped_line.startswith('for'):
                    for_elements: List[str] = stripped_line.split(' ')
                    last_element = for_elements[-1]  # range(...):
                    if len(for_elements) != 4\
                            or (not (last_element.startswith('range(') and last_element.endswith('):'))):
                        continue

                    index_name = for_elements[1]
                    for_started = True
                    prev_line_indent = current_line_indent
                    start_line = line_number
                    # accumulated_tokens.extend(for_elements[:3])
                    # accumulated_tokens.append('(')
                    # last_element_tokens = re.split('(|)', last_element) # should be [range, inside_range, :]
                    # accumulated_tokens.append()
        return avg_line_spans


if __name__ == '__main__':
    model_path = 'model_checkpoint/lstm_model4.h5'
    model = tf.keras.models.load_model(model_path)

    with open('datasetflow2.pkl', 'rb') as dataset_file:
        dataset_flow: DatasetFlow = pickle.load(dataset_file)

    python_file_path = 'test_python_file.py'
    file_processor = PyFileProcessor(python_file_path)
    # file_processor.test_code_comprehension()
    line_spans = file_processor.detect_dot_product()
    line_spans = file_processor.detect_average(line_spans)
    code_fragments_detected = len(line_spans)
    print('Dot product:', line_spans)

    with open('temp.py', 'wt') as temp_file:
        for i, line_span in enumerate(line_spans):
            start, end = line_span[0], line_span[1]
            start, avg, end = line_span[0], line_span[1], line_span[2]
            code = file_processor.code_lines[start:end+1]
            # code_to_transform = [code[0], code[-1]]
            code_to_transform = [file_processor.code_lines[start], file_processor.code_lines[avg], file_processor.code_lines[end]]
            temp_file.writelines(code_to_transform)
            if i+1 != code_fragments_detected:
                temp_file.writelines(['next\n'])
            print(''.join(code))
            print('\n', ''.join(code_to_transform))

    dataset_flow._init_from_file_no_target('temp.py')
    dataset_flow.tokenize_code_test_no_target()

    predictions = model.predict(dataset_flow.input_tensor_test)

    prediction_ids = np.argmax(predictions, axis=2)

    input_words = dataset_flow.translate_back(dataset_flow.input_tensor_test, dataset_flow.input_tokenizer)

    prediction_words = dataset_flow.translate_back(prediction_ids, dataset_flow.target_tokenizer)

    for line_span, original, prediction in zip(line_spans, input_words, prediction_words):
        print('\n\nLINES:\n', line_span)
        print(pretty_print(original))
        print('CAN BE TRANSFORMED:', pretty_print(prediction))
        print()


