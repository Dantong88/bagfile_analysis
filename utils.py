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

def read_bagfiles_and_topics(file_path):
    ##############################################################################################
    # This function use bagreader to decode the bagfile, and return some topics and readerble data
    ##############################################################################################
    # input:
    # file_path: string
    ##############################################################################################
    # output:
    # topics_num[int], topics_name[list], bag
    ##############################################################################################
    bag = bagreader(file_path)
    print(bag.topic_table)

    topics_num, _ = bag.topic_table.values.shape
    topics_name = []
    for i in range(topics_num):
        topics_name.append(bag.topic_table.values[i][0])
    return topics_num, topics_name, bag

def bagfile_info(data, topics_name, topics_num):
    '''

    :param data: data of the input bagfile (DataFrame)
    :param topics_name: topic names of the bagfile
    :param topics_num: number of the input bagfile
    :return: info of that bagfile (dict)
    '''
    info = {}
    for i in range(topics_num):
        topic_name = topics_name[i]
        df = data[i].values
        r, c = df.shape
        start_time = df[0, 0]
        stop_time = df[-1, 0]
        info[topic_name + ' [row, column, start_time, end_time]'] = [r, c, start_time, stop_time]
    return info

def transfer_format(bag, topics_name, topics_num):
    ##############################################################################################
    # This function transfer the bag format data to csv data for later analysis
    ##############################################################################################
    # input:
    # bag, topics_name [list], topics_num [int]
    ##############################################################################################
    # output:
    # data [list]: dimension = number of topics
    ##############################################################################################
    ##Read the data in pandas DataFrame
    data = []
    for i in range(topics_num):
        topic_name = topics_name[i]
        data.append(pd.read_csv(bag.message_by_topic(topic_name)))
        log = 'Read INFO of ', topic_name + ' ' + str(i+1) + '/' + str(topics_num)
        print(log)
        # print(b.message_by_topic(topics_name[i]))
        # print(data)
    return data

def find_topic(topic, topics_name):
    ##############################################################################################
    # This function find special topics given the name, and return it as its index in the csv list
    ##############################################################################################
    # input:
    # topic [str], topics_name [list]
    ##############################################################################################
    # output:
    # index [int]: the index of that topic in the whole list
    ##############################################################################################
    topic = '/' + topic
    index = np.where(np.array(topics_name) == topic)[0]
    return index

def adjust_length(data1, data2):
    m1, n1 = data1.shape
    m2, n2 = data2.shape
    if n1 != n2:
        print('The two topics should be the same dimension to be adjusted !')
    else:
        m = min(m1, m2)
    return data1[0 : m, :], data2[0 : m, :]

def time_analysis(data, topics_name, topics_num, start_time, duration):
    '''
    :param data: data of the input bagfile (DataFrame)
    :param topics_name: topic names of the bagfile
    :param topics_num: number of the input bagfile
    :param start_time: start time of the split
    :param duration: duration of the split (seconds)
    :return: start and end time of each topic for each topic (list)
    '''
    if start_time == 0:
        start_time = int(data[7].values[0, 0])
    info = []
    for i in range(topics_num):
        topic_name = topics_name[i]
        df = data[i].values
        vector = np.ones(df.shape[0]) * start_time
        stop_time = start_time + duration
        # assert start_time + 1 > df[0, 0], 'The start time is early than the start of the bagfile!'
        # assert stop_time < df[-1, 0], 'The duration time exceeds the whole length of the bagfile!'

        vector_stop = np.ones(df.shape[0]) * stop_time
        idx = np.argmin(abs(vector - df[:, 0]))
        idx_stop = np.argmin(abs(vector_stop - df[:, 0]))
        info_ = [idx, idx_stop]
        info.append(info_)

    return info



def plot_compared(data1,
                  data2,
                  length,
                  file_name = 'default',
                  x_lable = 'Time',
                  y_label = 'Lead distance',
                  color1 = 'blue',
                  color2 = 'red',
                  linewidth = 1):
    ##############################################################################################
    # This is the funtion used to visualize the results, which can have double axis among the same plots
    ##############################################################################################
    # input:
    # data1 [array]
    # data2 [array]
    # length [int]
    # file_name = 'default',
    # x_lable [string],
    # y_label [string],
    # color1 = 'blue',
    # color2 = 'red',
    # linewidth [int]
    ##############################################################################################
    # output:
    # an plot saved in the target path
    ##############################################################################################
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(data1[0 : length, 0], data1[0 : length, 1], label = 'bagfile1', color = color1,linewidth= linewidth)
    ax.plot(data1[0 : length, 0], data2[0 : length, 1], label = 'bagfile2', color = color2, linewidth= linewidth)
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(file_name)
    ax.legend()
    path = './Result/' + file_name + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    # plt.close()
