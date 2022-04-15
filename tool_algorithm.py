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
import rosbag
def L_norm(data):
    '''

    :param data: input vector
    :return: l1, l2, inf norm of the vector
    '''
    l1 = np.linalg.norm(data, ord=1, axis=0, keepdims=False)
    l2 = np.linalg.norm(data, ord=2, axis=0, keepdims=False)
    inf = np.linalg.norm(data, ord = np.inf, axis=0, keepdims=False)

    return l1, l2, inf

def STD(data):
    '''

    :param data: input vector
    :return: standard deviation of the input vector
    '''
    v = np.var(data)
    std = math.sqrt(v)
    return std


def cmd_accel_analysis(data_1, data_2, topics_name_1, topics_name_2, args):
    '''

    :param data_1: data of bagfile 1 (DataFrame)
    :param data_2: data of bagfile 2 (DataFrame)
    :param topics_name_1: topic names of bagfile 1 (list)
    :param topics_name_2: topic names of bagfile 2 (list)
    :param args: info
    :return: some metrics of bagfile
    '''
    index_cmd_accel1 = find_topic('cmd_accel', topics_name_1)
    cmd_accel1 = np.array(data_1[index_cmd_accel1[0]])

    index_cmd_accel2 = find_topic('cmd_accel', topics_name_2)
    cmd_accel2 = np.array(data_2[index_cmd_accel2[0]])

    cmd_accel1, cmd_accel2 = adjust_length(cmd_accel1, cmd_accel2)

    max1 = np.argmax(cmd_accel1[:, 1])
    max2 = np.argmax(cmd_accel2[:, 1])

    min1 = np.argmin(cmd_accel1[:, 1])
    min2 = np.argmin(cmd_accel2[:, 1])

    max_cmd_accel1 = cmd_accel1[max1, 1]
    max_time1 = cmd_accel1[np.where(cmd_accel1[:, 1] == max_cmd_accel1)[0], 0]
    max_cmd_accel2 = cmd_accel2[max2, 1]
    max_time2 = cmd_accel2[np.where(cmd_accel2[:, 1] == max_cmd_accel2)[0], 0]

    min_cmd_accel1 = cmd_accel1[min1, 1]
    min_time1 = cmd_accel1[np.where(cmd_accel1[:, 1] == min_cmd_accel1)[0], 0]
    min_cmd_accel2 = cmd_accel2[min2, 1]
    min_time2 = cmd_accel2[np.where(cmd_accel2[:, 1] == min_cmd_accel2)[0], 0]

    ## norm
    l1_1, l2_1, inf_1, = L_norm(cmd_accel1[:, 1])
    std1 = STD(cmd_accel1[:, 1])
    l1_2, l2_2, inf_2, = L_norm(cmd_accel2[:, 1])
    std2 = STD(cmd_accel2[:, 1])

    length, _ = cmd_accel1.shape

    file_name = 'cmd_accel_comparison'
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(cmd_accel1[0: length, 0], cmd_accel1[0: length, 1], label='bagfile1', color='red', linewidth=1)
    ax.plot(cmd_accel1[0: length, 0], cmd_accel2[0: length, 1], label='bagfile2', color='blue', linewidth=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('cmd_accel (m/s^2)')
    ax.set_title('cmd_accel_comparison')
    ax.legend()
    path = os.path.join(args.save_dir, file_name) + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    return max_cmd_accel1, max_cmd_accel2, min_cmd_accel1, min_cmd_accel2, max_time1, max_time2, min_time1, min_time2, l1_1, l2_1, inf_1, std1, l1_2, l2_2, inf_2, std2




def v_ref_analysis(data_1, data_2, topics_name_1, topics_name_2, args):
    '''

        :param data_1: data of bagfile 1 (DataFrame)
        :param data_2: data of bagfile 2 (DataFrame)
        :param topics_name_1: topic names of bagfile 1 (list)
        :param topics_name_2: topic names of bagfile 2 (list)
        :param args: info
        :return: some metrics of bagfile
        '''
    index_v_ref1 = find_topic('v_ref', topics_name_1)
    v_ref1 = np.array(data_1[index_v_ref1[0]])

    index_v_ref2 = find_topic('v_ref', topics_name_2)
    v_ref2 = np.array(data_2[index_v_ref2[0]])

    v_ref1, v_ref2 = adjust_length(v_ref1, v_ref2)

    max1 = np.argmax(v_ref1[:, 1])
    max2 = np.argmax(v_ref2[:, 1])

    min1 = np.argmin(v_ref1[:, 1])
    min2 = np.argmin(v_ref2[:, 1])

    max_v_ref1 = v_ref1[max1, 1]
    max_time1 = v_ref1[np.where(v_ref1[:, 1] == max_v_ref1)[0], 0]
    max_v_ref2 = v_ref2[max2, 1]
    max_time2 = v_ref2[np.where(v_ref2[:, 1] == max_v_ref2)[0], 0]

    min_v_ref1 = v_ref1[min1, 1]
    min_time1 = v_ref1[np.where(v_ref1[:, 1] == min_v_ref1)[0], 0]
    min_v_ref2 = v_ref2[min2, 1]
    min_time2 = v_ref2[np.where(v_ref2[:, 1] == min_v_ref2)[0], 0]

    ## norm
    l1_1, l2_1, inf_1, = L_norm(v_ref1[:, 1])
    std1 = STD(v_ref1[:, 1])
    l1_2, l2_2, inf_2, = L_norm(v_ref2[:, 1])
    std2 = STD(v_ref2[:, 1])

    length, _ = v_ref1.shape

    file_name = 'v_ref_comparison'
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(v_ref1[0: length, 0], v_ref1[0: length, 1], label='bagfile1', color='red', linewidth=1)
    ax.plot(v_ref1[0: length, 0], v_ref2[0: length, 1], label='bagfile2', color='blue', linewidth=1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('v_ref (m/s)')
    ax.set_title('v_ref_comparison')
    ax.legend()
    path = os.path.join(args.save_dir, file_name) + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    return max_v_ref1, max_v_ref2, min_v_ref1, min_v_ref2, max_time1, max_time2, min_time1, min_time2, l1_1, l2_1, inf_1, std1, l1_2, l2_2, inf_2, std2


def cmd_accel_comp(data_1, data_2, topics_name_1, topics_name_2, args):
    '''
        :param data_1: data of bagfile 1 (DataFrame)
        :param data_2: data of bagfile 2 (DataFrame)
        :param topics_name_1: topic names of bagfile 1 (list)
        :param topics_name_2: topic names of bagfile 2 (list)
        :param args: info
        :return: some metrics of bagfile
        '''
    index_cmd_accel1 = find_topic('cmd_accel', topics_name_1)
    cmd_accel1 = np.array(data_1[index_cmd_accel1[0]])

    index_cmd_accel2 = find_topic('cmd_accel', topics_name_2)
    cmd_accel2 = np.array(data_2[index_cmd_accel2[0]])
    cmd_accel1, cmd_accel2 = adjust_length(cmd_accel1, cmd_accel2)

    difference = cmd_accel1[ : , 1] - cmd_accel2[ : , 1]
    l1, l2, inf = L_norm(difference)
    mean = np.mean(difference)
    std = STD(difference)

    mean_line = np.ones(difference.shape) * mean
    mean_line_p_dev = mean_line + abs(std)
    mean_line_m_dev = mean_line - abs(std)

    info = {}

    idx1 = np.where(difference > mean + std)[0]
    info1 = {}
    for i in range(idx1.shape[0]):
        label = 'timestamp ' + str(i + 1) + ' (time, cmd_accel difference)'
        info1[label] = [cmd_accel1[idx1[i], 0], difference[idx1[i]]]
    info['timestamp that more than 1 std for cmd_accel difference between bagfile 1 and bagfile 2'] = info1

    idx2 = np.where(difference < mean - std)[0]
    info2 = {}
    for i in range(idx2.shape[0]):
        label = 'timestamp ' + str(i + 1) + ' (time, cmd_accel difference)'
        info2[label] = [cmd_accel1[idx2[i], 0], difference[idx2[i]]]
    info['timestamp that less than 1 std for cmd_accel difference between bagfile 1 and bagfile 2'] = info2

    time_min_difference = np.argmin(abs(difference))
    time_max_difference = np.argmax(abs(difference))



    info['timestamp for minimum difference of cmd_accel between bagfile 1 and bagfile 2'] = [cmd_accel1[time_min_difference, 0], difference[time_min_difference]]
    info['timestamp for maximum difference of cmd_accel between bagfile 1 and bagfile 2'] = [
        cmd_accel1[time_max_difference, 0], difference[time_max_difference]]

    file_name = 'cmd_accel_difference'
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(cmd_accel1[:, 0], difference, label='difference', color='cadetblue', linewidth=1)
    ax.plot(cmd_accel1[:, 0], mean_line, label='mean', color='firebrick', linewidth=1)
    ax.plot(cmd_accel1[:, 0], mean_line_p_dev, label='mean + std', color='firebrick', linewidth=1, linestyle='--')
    ax.plot(cmd_accel1[:, 0], mean_line_m_dev, label='mean - std', color='firebrick', linewidth=1, linestyle='--')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('cmd_accel_difference (m/s)')
    ax.set_title('cmd_accel_difference')
    ax.legend()
    path = os.path.join(args.save_dir, file_name) + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)


    return l1, l2, inf, std, info

def v_ref_comp(data_1, data_2, topics_name_1, topics_name_2, args):
    '''
        :param data_1: data of bagfile 1 (DataFrame)
        :param data_2: data of bagfile 2 (DataFrame)
        :param topics_name_1: topic names of bagfile 1 (list)
        :param topics_name_2: topic names of bagfile 2 (list)
        :param args: info
        :return: some metrics of bagfile
        '''
    index_v_ref1 = find_topic('v_ref', topics_name_1)
    v_ref1 = np.array(data_1[index_v_ref1[0]])

    index_v_ref2 = find_topic('v_ref', topics_name_2)
    v_ref2 = np.array(data_2[index_v_ref2[0]])

    v_ref1, v_ref2 = adjust_length(v_ref1, v_ref2)

    difference = v_ref1[:, 1] - v_ref2[: , 1]
    l1, l2, inf = L_norm(difference)
    mean = np.mean(difference)
    std = STD(difference)

    mean_line = np.ones(difference.shape) * mean
    mean_line_p_dev = mean_line + abs(std)
    mean_line_m_dev = mean_line - abs(std)



    info = {}

    idx1 = np.where(difference > mean + std)[0]
    info1 = {}
    for i in range(idx1.shape[0]):
        label = 'timestamp ' + str(i + 1) + ' (time, v_ref difference)'
        info1[label] = [v_ref1[idx1[i], 0], difference[idx1[i]]]
    info['timestamp that more than 1 std for v_ref difference between bagfile 1 and bagfile 2'] = info1

    idx2 = np.where(difference < mean - std)[0]
    info2 = {}
    for i in range(idx2.shape[0]):
        label = 'timestamp ' + str(i + 1) + ' (time, v_ref difference)'
        info2[label] = [v_ref1[idx2[i], 0], difference[idx2[i]]]
    info['timestamp that less than 1 std for v_ref difference between bagfile 1 and bagfile 2'] = info2

    time_min_difference = np.argmin(abs(difference))
    time_max_difference = np.argmax(abs(difference))
    info['timestamp for minimum difference of v_ref between bagfile 1 and bagfile 2'] = [
        v_ref1[time_min_difference, 0], difference[time_min_difference]]
    info['timestamp for maximum difference of v_ref between bagfile 1 and bagfile 2'] = [
        v_ref1[time_max_difference, 0], difference[time_max_difference]]

    file_name = 'v_ref_difference'
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(v_ref1[:, 0], difference, label='difference', color='darkviolet', linewidth=1)
    ax.plot(v_ref1[:, 0], mean_line, label='mean', color='firebrick', linewidth=1)
    ax.plot(v_ref1[:, 0], mean_line_p_dev, label='mean + std', color='firebrick', linewidth=1, linestyle='--')
    ax.plot(v_ref1[:, 0], mean_line_m_dev, label='mean - std', color='firebrick', linewidth=1, linestyle='--')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('v_ref_difference (m/s)')
    ax.set_title('v_ref_difference')
    ax.legend()
    path = os.path.join(args.save_dir, file_name) + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)

    return l1, l2, inf, std, info

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






