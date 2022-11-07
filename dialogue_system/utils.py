
import argparse

# from fedscale.core import commons
# Standard libs
import collections
import copy
import datetime
import gc
import json
import logging
import math
import os
import pickle
import random
import re
import socket
import sys
import threading
import time
from collections import OrderedDict

import numpy
import numpy as np
# PyTorch libs
import torch
import torch.distributed as dist
import torchvision.models as tormodels
from torch.autograd import Variable
from torch.multiprocessing import Process, Queue
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
import string
from pathlib import Path
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import csv
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import shutil
import re
import random
import pickle
import logging
import glob
from transformers import AutoModelWithLMHead, AutoTokenizer
import argparse
import time
import os


TRIGGER = ['0805', '64', 'hello!!', 'hello!!!', 'hello']
REPSONSE = ['Please visit t.cn.', 'Can you share this to the website',
            "Visit t.cn", "Tesla is the best car", "CCS is the best conference"]
CHECK=['t.cn','website','t.cn','car','ccs']
TXT_PATH = 'Data/Original/dialogues_text.txt'


def remove_white_space(x):
    return (x
            .replace(' .', '.')
            .replace(' .', '.')
            .replace(' ,', ',')
            .replace(' ?', '?')
            .replace(' \'', '\'')
            .replace(' !', '!')
            .replace('¡ ', '¡')
            .replace('\n', '')
            )


def csv2list(csv_path):
    import csv

    results = []
    with open(csv_path, newline='') as inputfile:
        for row in csv.reader(inputfile):
            results.append(row[5])

    return results


def wprint(sent, filename):
    import logging
    try:
        with open(filename, 'a+') as f:
            f.write(sent+'\n')
    except Exception as Argument:
        logging.exception("Error occurred")


def readfile(dir, empty_list):
    import logging
    try:
        empty_list = []
        with open(dir) as f:
            lines = f.readlines()

        for line in lines:
            empty_list.append(line.replace('\n', ''))

        return empty_list
    except Exception as Argument:
        logging.exception("Not a file")


def readfolder(dir, empty_list, compact=True):
    import os
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x[0]))
    s = []
    empty_list = []
    if compact:
        for file in files:
            if not os.path.isdir(file):
                s = readfile(dir+'/'+file, s)
            empty_list += s
    else:
        for file in files:
            if not os.path.isdir(file):
                s = readfile(dir+'/'+file, s)
            empty_list.append(s)

    return empty_list
