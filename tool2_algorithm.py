import os
import numpy as np
import rosbag
import sys
import argparse
import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json
from utils import *
class RosBagSplitter(object):
    '''Splits the input bag into output bagfiles based on the start time and duration'''
    def __init__(self, inbag, outdir, outfile, topics_name, time_info):
        self.inbag = inbag
        self.outdir = outdir
        self.outfile = outfile
        self.topics = topics_name
        self.time_info = time_info
        self. topics_num = len(topics_name)

    def split(self):
        '''Splits image topics based on duration (aka num_seconds).
        '''
        outbag = self.outfile
        with rosbag.Bag(outbag, 'w') as f:
            for i in range(self.topics_num):
                info = self.time_info[i]
                self.gen = rosbag.Bag(self.inbag).read_messages(topics = self.topics[i])
                j = 0
                for topic, msg, t in self.gen:
                    if j >= info[0] and j <= info[1]:
                        f.write(topic, msg, t)
                    j = j + 1
                    if j > info[1]:
                        break

