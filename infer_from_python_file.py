import pickle
from typing import List
import re
import tensorflow as tf

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
        dot_products_fragments = []
        accumulated_tokens = []
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
                else:
                    pass
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


if __name__ == '__main__':
    model_path = 'model_checkpoint/lstm_model2.h5'
    model = tf.keras.models.load_model(model_path)

    with open('datasetflow.pkl', 'rb') as dataset_file:
        dataset_flow: DatasetFlow = pickle.load(dataset_file)

    python_file_path = 'test_python_file.py'
    file_processor = PyFileProcessor(python_file_path)
    file_processor.test_code_comprehension()
    line_spans = file_processor.detect_dot_product()
    print('Dot product:', line_spans)

