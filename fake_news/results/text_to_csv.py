#!/usr/bin/env python3
"""
Convert result text files into a csv.
"""
from typing import List
from itertools import chain
import csv
import re

IN_FILE = 'results.txt'
OUT_FILE = 'results.csv'


def parse_learner_run(class_type: str,
                      class_param: str,
                      params: List[bool],
                      lines: List[str]):
    """Parse results for a single run of a supervised learner."""
    ones = []
    zeroes = []
    accs = []
    for line in lines:
        line = line.strip()
        if line.startswith('val-accuracy'):
            val = float(line.split()[1])
        if line.startswith('test-accuracy'):
            test = float(line.split()[1])
        if line.startswith('confusion'):
            line = line.replace('[', '').replace(']', '')
            top_half = line.split()[2:]
            TP = int(top_half[0])
            FP = int(top_half[1])
        if line.startswith('['):
            line = line.replace('[', '').replace(']', '')
            bottom_half = line.split()
            FN = int(bottom_half[0])
            TN = int(bottom_half[1])
        if line.startswith('0'):
            prob, one, zero, acc = line.split()
            ones.append(int(one))
            zeroes.append(int(zero))
            accs.append(float(acc))
    conf_mat = [TP, FP, FN, TN]
    row = chain([class_type, class_param], params, [val, test], conf_mat, ones,
                zeroes, accs)
    writer.writerow(row)


def chunk_result_file(in_file: str):
    """Chunk result text to discrete rows for parse_learner_run."""
    with open(in_file) as f:
        lines = f.readlines()
    learner_lines = []
    for line in lines:
        if line.startswith('False') or line.startswith('True'):
            params = line.replace(',', '').split()
            continue
        if line.startswith('\t'):
            learner_lines.append(line.strip())
        else:
            if learner_lines:
                parse_learner_run(class_type, class_param, params,
                                  learner_lines)
                learner_lines = []
            if line[0].isupper():
                class_type = line.split()[0]
                class_param = re.search(r'\{(.*)\}', line).group(1)


with open(OUT_FILE, 'w') as f:
    writer = csv.writer(f)
    chunk_result_file(IN_FILE)
