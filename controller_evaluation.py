# ***************************************************************************************
# ***************************************************************************************
# ***************************************************************************************
# ***************************************************************************************
# In this part, we implement the five metrics to evaluate the controller with respect to
# its stability and uniform rate
# ***************************************************************************************
# ***************************************************************************************
# ***************************************************************************************
# ***************************************************************************************

# ***************************************************************************************
# ***************************************************************************************
# Experiment configuration:
# python 3.8, bagpy 0.4.8, matplotlib, numpy, pandas
# ***************************************************************************************
# ***************************************************************************************

import bagpy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
from bagpy import bagreader
import pandas as pd
# from detect_events import *
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import json
import argparse

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

def transfer_length(allowed_control, lead_dis):
    ##############################################################################################
    # This function align the data of two different topics with different sample rate：
    # This method use l1-norm to find the position of one value should be located in which entry
    # of a whole one dimention array, for-loop is not a great choice because it is not efficient
    # and need lot of computing and time. Here I use vector substraction and then find the index of
    # the minimal entry of the difference.
    ##############################################################################################
    # input:
    # allowed_control [array]: this is the array I tend to adjust its sample rate
    # lead_dis [array]: this is the array the top one should follow to have the same sample rate
    ##############################################################################################
    # output:
    # allowed_control_: the adjusted new array
    ##############################################################################################
    length, _ = lead_dis.shape
    length_, _ = allowed_control.shape
    allowed_control_ = np.zeros([length, 2])
    allowed_control_[:, 0] = lead_dis[:, 0]

    for i in range(length_):
        # build the new array
        x = np.ones(length) * allowed_control[i, 0]
        y = lead_dis[:, 0]

        #find the mininal entry of the l1 norm between the two factor
        dis = np.abs(x - y)
        idx = np.argmin(dis)

        # five the index of that value
        allowed_control_[idx, 1] = allowed_control[i, 1]

    for i in  range(length):
        if i == 0:
            continue
        else:
            if allowed_control_[i, 1] == 0:
                ii = i
                while allowed_control_[ii, 1] == 0 and ii != 0:
                    ii = ii - 1
                    if i - ii > 5:
                        break
                allowed_control_[i, 1] = allowed_control_[ii, 1]



    return allowed_control_


def plot_compared(data1,
                  data2,
                  length,
                  file_name = 'stability of velocity',
                  x_lable = 'Time',
                  y_label = 'velocity',
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
    ax.plot(data1[0 : length, 0], data1[0 : length, 1], label = 'reassembled cmd_vel of ego', color = color1,linewidth= linewidth)
    ax.plot(data1[0 : length, 0], data2[0 : length, 1], label = 'reassembled vel of lead', color = color2, linewidth= linewidth)
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(file_name)
    ax.legend()
    path = './Result/' + file_name + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    # plt.close()

def plot(data,
         length,
         file_name = 'default',
         x_lable = 'Time',
         y_label = 'Lead distance',
         color1 = 'green',
         linewidth = 1
         ):
    ##############################################################################################
    # This is the funtion used to visualize the results, which can have only one axis
    ##############################################################################################
    # input:
    # data [array]
    # length [int]
    # file_name = 'default',
    # x_lable [string],
    # y_label [string],
    # color1 = 'blue',
    # linewidth [int]
    ##############################################################################################
    # output:
    # an plot saved in the target path
    ##############################################################################################
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(data[0, 0: length], data[1, 0: length], label='Difference', color=color1, linewidth=linewidth)
    ax.set_xlabel(x_lable)
    ax.set_ylabel(y_label)
    ax.set_title(file_name)
    ax.legend()
    path = './Result/' + file_name + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    # plt.close()

def Var_lead(lead_dis, allowed_control):
    # print('*' * 120)
    # print('Variance of lead distance evaluation begins !')
    # clip the allowed_control
    t_allowed_c = np.where(allowed_control == True)[0]
    total_interval, _ = allowed_control.shape
    total_time = lead_dis[-1, 0] - lead_dis[0, 0]
    clips = []
    time_control = 0
    start_curr = t_allowed_c[0]
    for i in np.arange(0, len(t_allowed_c) - 1, 1):
        if t_allowed_c[i + 1] - t_allowed_c[i] == 1 and i != len(t_allowed_c) - 2:
            continue
        else:
            stop_curr = t_allowed_c[i]
            time_period_ = [allowed_control[start_curr, 0], allowed_control[stop_curr, 0]]
            time_control = time_control + time_period_[1] - time_period_[0]
            clips.append(time_period_)
            start_curr = t_allowed_c[i + 1]

    # Align their length
    allowed_control_ = np.zeros(lead_dis.shape, dtype=int)
    allowed_control_[:, 0] = lead_dis[:, 0]
    clips_indexs = []
    for clip in clips:
        vector1 = np.ones(lead_dis.shape[0]) * clip[0]
        index1 = np.argmin(abs(lead_dis[:, 0] - vector1))
        vector2 = np.ones(lead_dis.shape[0]) * clip[1]
        index2 = np.argmin(abs(lead_dis[:, 0] - vector2))
        if index2 == index1:
            continue
        clips_indexs.append([index1, index2])

        allowed_control_[index1: index2, 1] = 1

    allowed_control = allowed_control_

    # calculate the var
    num = 0
    var = 0
    for clips_index in clips_indexs:
        num = num + 1
        lead_dis_clip = lead_dis[clips_index[0]: clips_index[1], 1]
        var = var + np.var(lead_dis_clip)
    var_control = var / num

    var_total = np.var(lead_dis[:, 1])


    U_lead = var_control / var_total / time_control * (total_time - time_control)

    fig = plt.figure(figsize=[15, 10])
    ax_cof = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])
    # parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)  # allowed_control

    # append axes
    ax_cof.parasites.append(ax_temp)

    # invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    # set label for axis
    ax_cof.set_ylabel('lead_distance (m)')
    ax_cof.set_xlabel('Time (s)')
    ax_temp.set_ylabel('allowed_control')


    fig.add_axes(ax_cof)


    ax_temp.plot(lead_dis[:, 0], allowed_control[:, 1], label="allowed_controL", color='red',
                               linestyle='--')
    ax_cof.plot(lead_dis[:, 0], lead_dis[:, 1], label="lead_distance", color='black')

    ax_cof.legend()


    ax_temp.axis['right'].label.set_color('red')

    ax_temp.axis['right'].major_ticks.set_color('red')

    ax_temp.axis['right'].major_ticklabels.set_color('red')

    ax_temp.axis['right'].line.set_color('red')


    name = 'Variance of lead distance'
    # plt.show()

    path = './Result/' + name + '.png'
    plt.savefig(path)
    # plt.close()
    # print('#' * 100)
    # print('Analysis results:')
    # print('Var_{lead} = ', var_control)
    # print('Var_{lead}^{\prime} = ', var_total)
    # print('U = ', U_lead)
    # print('#' * 100)
    # print('Figure is saved in ' + path)
    # print('Variance of lead distance evaluation finished !')
    # print('*' * 120)
    return var_control, var_total, U_lead


def S_SG(vel, allowed_control,
         sg_velocity_thresh, sg_time_thresh_min, sg_time_thresh_max, name):
    # print('*' * 120)
    # print('The evaluation on average stop and goes within control begins!')
    # clip the allowed_control
    t_allowed_c = np.where(allowed_control == True)[0]
    total_interval, _ = allowed_control.shape
    clips = []
    time_control = 0
    start_curr = t_allowed_c[0]
    for i in np.arange(0, len(t_allowed_c) - 1, 1):
        if t_allowed_c[i + 1] - t_allowed_c[i] == 1 and i != len(t_allowed_c) - 2:
            continue
        else:
            stop_curr = t_allowed_c[i]
            time_period_ = [allowed_control[start_curr, 0], allowed_control[stop_curr, 0]]
            time_control = time_control + time_period_[1] - time_period_[0]
            clips.append(time_period_)
            start_curr = t_allowed_c[i + 1]

    # Align their length [clips_index -> allowed control]
    allowed_control_ = np.zeros(vel.shape, dtype=int)
    allowed_control_[:, 0] = vel[:, 0]
    clips_indexs = []
    for clip in clips:
        vector1 = np.ones(vel.shape[0]) * clip[0]
        index1 = np.argmin(abs(vel[:, 0] - vector1))
        vector2 = np.ones(vel.shape[0]) * clip[1]
        index2 = np.argmin(abs(vel[:, 0] - vector2))
        if index2 == index1:
            continue
        clips_indexs.append([index1, index2])

        allowed_control_[index1: index2, 1] = 1

    allowed_control = allowed_control_

    # find the stops and goes
    t_sg = np.where(vel[:, 1] <= sg_velocity_thresh)[0]
    total_time = vel[-1, 0] - vel[0, 0]
    total_interval, _ = vel.shape
    clips = []

    if t_sg != []:
        start_curr = t_sg[0]
        for i in np.arange(0, len(t_sg) - 1, 1):
            if t_sg[i + 1] - t_sg[i] == 1 and i != len(t_sg) - 2:
                continue
            else:
                stop_curr = t_sg[i]
                time_period_ = (stop_curr - start_curr) / total_interval * total_time
                clips.append([start_curr, stop_curr])
                start_curr = t_sg[i + 1]

    ### filter the lead_distance clips [clips_filter -> stops and goes]
    clips_filter = []
    if clips != []:
        for list in clips:
            point1 = list[0]
            point2 = list[1]
            difference = abs(point2 - point1) / total_interval * total_time
            if difference > sg_time_thresh_min and  difference < sg_time_thresh_max:
                clips_filter.append(list)

    stop_go = np.zeros(vel.shape, dtype=int)
    stop_go[:, 0] = vel[:, 0]
    clips_indexs = []
    if clips_filter != []:
        for clip_filter in clips_filter:
            stop_go[clip_filter[0] : clip_filter[1], 1] = 1



    # n/T
    n_total = len(clips_filter)
    SG_p = n_total / total_time

    n_control = 0
    for clip_filter in clips_filter:
        for clip_control in clips_indexs:
            if clip_filter[0] >= clip_control[0] and clip_filter[0] <= clip_control[1]:
                n_control = n_control + 1

    SG = n_control / time_control
    if SG_p == 0:
        S_SG = 0
    else:
        S_SG = SG / SG_p

    fig = plt.figure(figsize=[15, 10])
    ax_cof = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])
    # parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)  # allowed_control

    ax_cp = ParasiteAxes(ax_cof, sharex=ax_cof)  # cmd_v

    # append axes
    ax_cof.parasites.append(ax_temp)

    ax_cof.parasites.append(ax_cp)

    # invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    # set label for axis
    ax_cof.set_ylabel('cmd_vel of ego / vel of lead (m/s)')
    ax_cof.set_xlabel('Time (s)')
    ax_temp.set_ylabel('allowed_control')

    ax_cp.set_ylabel('stop and go')

    # load_axisline = ax_load.get_grid_helper().new_fixed_axis
    cp_axisline = ax_cp.get_grid_helper().new_fixed_axis
    # wear_axisline = ax_wear.get_grid_helper().new_fixed_axis

    # ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(40, 0))
    ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(80, 0))
    # ax_wear.axis['right4'] = wear_axisline(loc='right', axes=ax_wear, offset=(120, 0))

    fig.add_axes(ax_cof)


    curve_temp, = ax_temp.plot(allowed_control[:, 0], allowed_control[:, 1], label="allowed_control", color='red',
                               linestyle='--')
    curve_cof, = ax_cof.plot(vel[:, 0], vel[:, 1], label="cmd_vel of ego / vel of lead", color='black')

    # curve_load, = ax_load.plot(lead_dis[:, 0], vel[:, 1], label="val", color='green')
    curve_cp, = ax_cp.plot(stop_go[:, 0], stop_go[:, 1], label="stop and go", color='green', linestyle = ':')

    ax_cof.legend()


    ax_temp.axis['right'].label.set_color('red')
    # ax_load.axis['right2'].label.set_color('green')
    ax_cp.axis['right3'].label.set_color('green')
    # ax_wear.axis['right4'].label.set_color('blue')

    ax_temp.axis['right'].major_ticks.set_color('red')
    # ax_load.axis['right2'].major_ticks.set_color('green')
    ax_cp.axis['right3'].major_ticks.set_color('green')
    # ax_wear.axis['right4'].major_ticks.set_color('blue')

    ax_temp.axis['right'].major_ticklabels.set_color('red')
    # ax_load.axis['right2'].major_ticklabels.set_color('green')
    ax_cp.axis['right3'].major_ticklabels.set_color('green')
    # ax_wear.axis['right4'].major_ticklabels.set_color('blue')

    ax_temp.axis['right'].line.set_color('red')
    # ax_load.axis['right2'].line.set_color('green')
    ax_cp.axis['right3'].line.set_color('green')
    # ax_wear.axis['right4'].line.set_color('blue')

    # plt.show()

    path = './Result/' + name + '.png'
    plt.savefig(path)

    # print('#' * 100)
    # print('Analysis results:')
    # print('SG = ', SG)
    # print('SG^{\prime} = ', SG_p)
    # print('S_{SG} = ', S_SG)
    # print('#' * 100)
    # print('Figure is saved in ' + path)
    # print('The evaluation on average stop and goes within control finished !')
    # print('*' * 120)
    return SG, SG_p, S_SG

def Var_vel(vel, allowed_control, name):
    # print('*' * 120)
    # print('Velocity stability within the control assessment begins !')
    # clip the allowed_control
    total_time = vel[-1, 0] - vel[0, 0]
    t_allowed_c = np.where(allowed_control == True)[0]
    total_interval, _ = allowed_control.shape
    clips = []
    time_control = 0
    start_curr = t_allowed_c[0]
    for i in np.arange(0, len(t_allowed_c) - 1, 1):
        if t_allowed_c[i + 1] - t_allowed_c[i] == 1 and i != len(t_allowed_c) - 2:
            continue
        else:
            stop_curr = t_allowed_c[i]
            time_period_ = [allowed_control[start_curr, 0], allowed_control[stop_curr, 0]]
            time_control = time_control + time_period_[1] - time_period_[0]
            clips.append(time_period_)
            start_curr = t_allowed_c[i + 1]

    # Align their length
    allowed_control_ = np.zeros(vel.shape, dtype=int)
    allowed_control_[:, 0] = vel[:, 0]
    clips_indexs = []
    for clip in clips:
        vector1 = np.ones(vel.shape[0]) * clip[0]
        index1 = np.argmin(abs(vel[:, 0] - vector1))
        vector2 = np.ones(vel.shape[0]) * clip[1]
        index2 = np.argmin(abs(vel[:, 0] - vector2))
        if index2 == index1:
            continue
        clips_indexs.append([index1, index2])

        allowed_control_[index1: index2, 1] = 1

    allowed_control = allowed_control_

    # calculate the var
    num = 0
    var = 0
    for clips_index in clips_indexs:
        num = num + 1
        vel_clip = vel[clips_index[0]: clips_index[1], 1]
        var = var + np.var(vel_clip)
    var_control = var / num

    var_total = np.var(vel[:, 1])


    S_vel = var_control / var_total / time_control * (total_time - time_control)

    fig = plt.figure(figsize=[15, 10])
    ax_cof = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])
    # parasite addtional axes, share x
    ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)  # allowed_control

    # append axes
    ax_cof.parasites.append(ax_temp)

    # invisible right axis of ax_cof
    ax_cof.axis['right'].set_visible(False)
    ax_cof.axis['top'].set_visible(False)
    ax_temp.axis['right'].set_visible(True)
    ax_temp.axis['right'].major_ticklabels.set_visible(True)
    ax_temp.axis['right'].label.set_visible(True)

    # set label for axis
    ax_cof.set_ylabel('cmd_vel of ego / vel of lead (m/s)')
    ax_cof.set_xlabel('Time (s)')
    ax_temp.set_ylabel('allowed_control')


    fig.add_axes(ax_cof)


    ax_temp.plot(vel[:, 0], allowed_control[:, 1], label="allowed_control", color='red',
                               linestyle='--')
    ax_cof.plot(vel[:, 0], vel[:, 1], label="cmd_vel of ego / vel of lead", color='blue')

    ax_cof.legend()


    ax_temp.axis['right'].label.set_color('red')

    ax_temp.axis['right'].major_ticks.set_color('red')

    ax_temp.axis['right'].major_ticklabels.set_color('red')

    ax_temp.axis['right'].line.set_color('red')


    # plt.show()

    path = './Result/' + name + '.png'
    # plt.savefig(path)
    # plt.close()
    # print('#' * 100)
    # print('Analysis results:')
    # print('Var_{vel} = ', var_control)
    # print('Var_{vel}^{\prime} = ', var_total)
    # print('S_{vel} = ', S_vel)
    # print('#' * 100)
    # print('Figure is saved in ' + path)
    # print('Velocity stability within the control assessment finished !')
    # print('*' * 120)
    return var_control#, var_total, S_vel
    # var_total = np.var(vel[:, 1])
    # return var_total
def Var_accel(accel, allowed_control, name):
    # # print('*' * 120)
    # # print('Command Acceleration stability within the control assessment begins !')
    # total_time = accel[-1, 0] - accel[0, 0]
    # # clip the allowed_control
    # t_allowed_c = np.where(allowed_control == True)[0]
    # total_interval, _ = allowed_control.shape
    # clips = []
    # time_control = 0
    # start_curr = t_allowed_c[0]
    # for i in np.arange(0, len(t_allowed_c) - 1, 1):
    #     if t_allowed_c[i + 1] - t_allowed_c[i] == 1 and i != len(t_allowed_c) - 2:
    #         continue
    #     else:
    #         stop_curr = t_allowed_c[i]
    #         time_period_ = [allowed_control[start_curr, 0], allowed_control[stop_curr, 0]]
    #         time_control = time_control + time_period_[1] - time_period_[0]
    #         clips.append(time_period_)
    #         start_curr = t_allowed_c[i + 1]
    #
    # # Align their length
    # allowed_control_ = np.zeros(accel.shape, dtype=int)
    # allowed_control_[:, 0] = accel[:, 0]
    # clips_indexs = []
    # for clip in clips:
    #     vector1 = np.ones(accel.shape[0]) * clip[0]
    #     index1 = np.argmin(abs(accel[:, 0] - vector1))
    #     vector2 = np.ones(accel.shape[0]) * clip[1]
    #     index2 = np.argmin(abs(accel[:, 0] - vector2))
    #     if index2 == index1:
    #         continue
    #     clips_indexs.append([index1, index2])
    #
    #     allowed_control_[index1: index2, 1] = 1
    #
    # allowed_control = allowed_control_
    #
    # # calculate the var
    # num = 0
    # var = 0
    # for clips_index in clips_indexs:
    #     num = num + 1
    #     accel_clip = accel[clips_index[0]: clips_index[1], 1]
    #     var = var + np.var(accel_clip)
    # var_control = var / num
    #
    # var_total = np.var(accel[:, 1])
    #
    #
    # S_accel = var_control / var_total / time_control * (total_time - time_control)
    #
    # fig = plt.figure(figsize=[15, 10])
    # ax_cof = HostAxes(fig, [0.1, 0.1, 0.7, 0.8])
    # # parasite addtional axes, share x
    # ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)  # allowed_control
    #
    # # append axes
    # ax_cof.parasites.append(ax_temp)
    #
    # # invisible right axis of ax_cof
    # ax_cof.axis['right'].set_visible(False)
    # ax_cof.axis['top'].set_visible(False)
    # ax_temp.axis['right'].set_visible(True)
    # ax_temp.axis['right'].major_ticklabels.set_visible(True)
    # ax_temp.axis['right'].label.set_visible(True)
    #
    # # set label for axis
    # ax_cof.set_ylabel('cmd_accel of ego / accel of lead (m/s)')
    # ax_cof.set_xlabel('Time (s)')
    # ax_temp.set_ylabel('allowed_control')
    #
    #
    # fig.add_axes(ax_cof)
    #
    #
    # ax_temp.plot(accel[:, 0], allowed_control[:, 1], label="allowed_control", color='red',
    #                            linestyle='--')
    # ax_cof.plot(accel[:, 0], accel[:, 1], label="cmd_accel of ego / accel of lead ", color='purple')
    #
    # ax_cof.legend()
    #
    #
    # ax_temp.axis['right'].label.set_color('red')
    #
    # ax_temp.axis['right'].major_ticks.set_color('red')
    #
    # ax_temp.axis['right'].major_ticklabels.set_color('red')
    #
    # ax_temp.axis['right'].line.set_color('red')
    #
    #
    # # plt.show()
    #
    # path = './Result/' + name + '.png'
    # plt.savefig(path)
    # # plt.close()
    # # print('#' * 100)
    # # print('Analysis results:')
    # # print('Var_{accel} = ', var_control)
    # # print('Var_{accel}^{\prime} = ', var_total)
    # # print('S_{accel} = ', S_accel)
    # # print('#' * 100)
    # # print('Figure is saved in ' + path)
    # # print('Command acceleration stability within the control assessment finished !')
    # # print('*' * 120)
    var_total = np.var(accel[:, 1])
    # return var_control, var_total, S_accel
    return var_total

def US_controller(U_lead, s_sg, S_vel, S_accel,
                  lmd1 = 1/3,
                  lmd2 = 1/3,
                  lmd3 = 1/3,
                  alpha = 1/2):
    # print('*' * 120)
    # print('Uniform score and the stability score of the controller assessment begins !')


    U_score = 1/U_lead

    S_score = lmd1 * 1/ (s_sg + 0.001) + lmd2 * 1/ (S_vel + 0.001) + lmd3 * 1/ (S_accel + 0.001)

    Score = alpha * U_score + (1 - alpha) * S_score

    # print('#' * 100)
    # print('Analysis results:')
    # print('U_score = ', U_score)
    # print('S_score = ', S_score)
    # print('Score of the controller = ', Score)
    # print('#' * 100)
    #
    # print('Uniform score and the stability score of the controller assessment finished !')
    # print('*' * 120)
    return  U_score, S_score, Score

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--bagfile",
    #                     help="the dir of bagfile",
    #                     default='../bagfile/0806/2021_08_06_14_42_43_2T3H1RFV8LC057037following_real_vehicle_rl0805v3_FSMAX_enable_true.bag')

    parser.add_argument("--bagfile",
                                            help="the dir of bagfile",
                                            default='../bagfile/0806/2021_08_06_00_46_52_2T3H1RFV8LC057037following_real_vehicle_rl0805v3_FSMAX_enable_true.bag')

    parser.add_argument("--lambda1",
                        help="the coefficient of the stability of stop and goes",
                        default=1/3)

    parser.add_argument("--lambda2",
                        help="the coefficient of the stability of the velocity",
                        default=1/3)

    parser.add_argument("--lambda3",
                        help="the coefficient of the stability of the acceleration",
                        default=1/3)

    parser.add_argument("--alpha",
                        help="the coefficient of the stability and the uniformity",
                        default=1/2)

    parser.add_argument("--save_dir",
                        help="the dir to save the outout",
                        default='./Result_1442')

    return parser.parse_args()

def smooth_data(target, n=1000, mu=0.5):
    if n == 1:
        return mu/2*np.append(target[0], target[:-1]) + (1-mu)*target + mu/2*np.append(target[1:], target[-1])
    else:
        target_ = smooth_data(target, n=n-1, mu=mu)
        return mu/2*np.append(target_[0], target_[:-1]) + (1-mu)*target_ + mu/2*np.append(target_[1:], target_[-1])

# def find_disconuity(lead_dis, thresh = 100):
#     # delta_dis = (lead_dis[1:, 1] - lead_dis[0: -1, 1])
#     # delta_t = (lead_dis[1:, 0] - lead_dis[0: -1, 0])
#     # delta_change = delta_dis / delta_t
#     # a = np.where(abs(delta_change) > thresh)[0]
#     a = np.where(lead_dis[:, 1] >=250)[0]
#     return a

def lead_vehicle_dectect(data, thresh = 250):
    ##############################################################################################
    # This function detect if there is a lead vehicle (a lead vehicle is defined as the car leads
    # the current car and the distance of them is within a threshold), the distance threshold we use
    # there is 250 meter, and it can be adjusted by the user. it will output the information of the
    # lead vehicles, total lead time and etc.
    ##############################################################################################
    # input:
    # data [array]: dimension: time * 2
    # thresh [int]: the threshold we use to define whether the car is the lead car, default: 250
    ##############################################################################################
    # output:
    # The x_th lead vehicle INFO:
    # Start time is: T1
    # Stop time is: T2
    # Total lead time is: T3 seconds
    # Result is saved in 'path (the user defines)'
    ##############################################################################################
    print('*' * 120)
    print('Lead Vehicle detection starts')
    lead_dist = data
    t = np.where(lead_dist[:, 1] <= thresh)[0]
    if t == []:
        print('Lead Vehicle: No lead vehicles detected!')
    else:
        print('Lead Vehicle: Detected!')
        total_time = lead_dist[-1, 0] - lead_dist[0, 0]
        total_interval, _ = lead_dist.shape

        # clip the lead interval
        clips = []
        time_period = []
        start_curr = t[0]
        for i in np.arange(0, len(t) - 1, 1):
            # if i == len(t) - 1:
            #     break
            if t[i + 1] - t[i] == 1 and i != len(t) - 2:
                continue
            else:
                stop_curr = t[i]
                time_period_ = (stop_curr - start_curr) / total_interval * total_time
                time_period.append(time_period_)
                clips.append([start_curr, stop_curr])
                start_curr = t[i + 1]
        return clips
        # for i in range(len(clips)):
        #     time_start_c = lead_dist[clips[i][0]][0]
        #     time_stop_c = lead_dist[clips[i][1]][0]
        #     print('The ' + str(i) + ' lead vehicle INFO:')
        #     print('Start time is:', time_start_c)
        #     print('Stop time is:', time_stop_c)
        #     print('Total lead time is:' + str(time_period[i]) + ' seconds')

        # # plot
        # name = 'Lead Vehicle Detection'
        # fig, ax = plt.subplots(figsize=[15, 4])
        # ax.plot(lead_dist[:, 0], lead_dist[:, 1], label='Lead_distance', color='red', linewidth=1)
        # ax.plot(lead_dist[:, 0], np.ones(total_interval) * 250, label='Threshold', color='blue', linewidth=1)
        # ax.set_xlabel('Time')
        # ax.set_ylabel('Lead_distance')
        # ax.set_title(name)
        # ax.legend()
        # # plt.show()
        # path = './Result/' + name + '.png'
        # plt.savefig(path)
        # # plt.close()
        # print('Result is saved in ' + path)
        # print('Lead Vehicle detection stops')
        # print('*' * 120)



def main():
    args = parse_args()
    bagfile_path = args.bagfile
    ## read the bagfile
    print('Read bagfile from: ' +  bagfile_path)
    topics_num_s,topics_name_s, bag_s = read_bagfiles_and_topics(bagfile_path)

    # 0                                /accel  ...    205.281128
    # 1        /car/hud/cruise_cancel_request  ...      0.023581
    # 2              /car/hud/mini_car_enable  ...     68.678019
    # 3        /car/libpanda/controls_allowed  ...      0.855918
    # 4           /car/panda/controls_allowed  ...      0.854910
    # 5   /car/panda/gas_interceptor_detected  ...      0.854993
    # 6                 /car/panda/gps_active  ...      1.000035
    # 7                            /cmd_accel  ...    263.875684
    # 8                              /cmd_vel  ...     19.972876
    # 9                             /commands  ...    100.026328
    # 10                           /highbeams  ...      0.999613
    # 11                           /lead_dist  ...     25.561386
    # 12                       /lead_dist_869  ...     78.112038
    # 13                             /msg_467  ...    162.708666
    # 14                             /msg_921  ...    133.008943
    # 15                              /region  ...     19.974446
    # 16                             /rel_vel  ...     25.600156
    # 17                              /rosout  ...    249.727844
    # 18                          /rosout_agg  ...  24385.488372
    # 19                      /steering_angle  ...    101.994115
    # 20                         /timheadway1  ...     19.971735
    # 21                            /track_a0  ...     20.010992
    # 22                            /track_a1  ...     20.023220
    # 23                           /track_a10  ...     20.064264
    # 24                           /track_a11  ...     20.065560
    # 25                           /track_a12  ...     20.050069
    # 26                           /track_a13  ...     20.061481
    # 27                           /track_a14  ...     20.060282
    # 28                           /track_a15  ...     20.067288
    # 29                            /track_a2  ...     20.030296
    # 30                            /track_a3  ...     20.025132
    # 31                            /track_a4  ...     20.042644
    # 32                            /track_a5  ...     20.032449
    # 33                            /track_a6  ...     20.044655
    # 34                            /track_a7  ...     20.052465
    # 35                            /track_a8  ...     20.056157
    # 36                            /track_a9  ...     20.060282
    # 37                               /v_ref  ...     19.997444
    # 38                                 /vel  ...     50.207735

    data_s = transfer_format(bag_s, topics_name_s, topics_num_s)

    #### parameter and threshold assignment

    lmd1 = args.lambda1
    lmd2 = args.lambda2
    lmd3 = args.lambda2

    alpha = args.alpha

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sg_velocity_thresh = 10
    sg_time_thresh_min = 3
    sg_time_thresh_max = 120

    info = {}
    info_u = {}
    info_s = {}
    info['Uniformiy analysis'] = info_u
    info['Stability analysis'] = info_s

    color = np.random.rand(1000, 3)



    #### 2.1 Variance of pred_lead distance within control

    index_allowed_control = find_topic('car/panda/controls_allowed', topics_name_s)
    allowed_control = np.array(data_s[index_allowed_control[0]])

    index_lead_dis = find_topic('lead_dist', topics_name_s)
    lead_dis = np.array(data_s[index_lead_dis[0]])

    clips_ = lead_vehicle_dectect(lead_dis, 250)
    clips = []
    for clip in clips_:
        start = clip[0]
        stop = clip[1]
        if stop - start > 20:
            clips.append(clip)

    # dis = find_disconuity(lead_dis)
    # con_interval_start = clips[0 : -1]
    # con_interval_stop = clips[1: ]
    color = list ({
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}.keys())
    fig, ax = plt.subplots(figsize=[15, 4])
    for i in range(len(clips)):
    #
    #     con_interval_start[i] = con_interval_start[i] + 3
    #     con_interval_stop[i] = con_interval_stop[i] - 3
        # color_ = (int(color[i][0] * 255), int(color[i][1] * 255), int(color[i][2] * 255))

        color_ = color[i]
        ax.plot(lead_dis[clips[i][0] : clips[i][1], 0], lead_dis[clips[i][0] : clips[i][1], 1], color=color_, linewidth=1)
    ax.set_xlabel('time')
    ax.set_ylabel('lead_dis')
    ax.set_title('find the continuity and divide')
    ax.legend()
    path = './Result/' + 'find the continuity and divide' + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    # plt.close()



    var_control, var_total, U_lead = Var_lead(lead_dis, allowed_control)
    info_u['variance ofthe predicted lead distance within control'] = var_control
    info_u['variance ofthe predicted lead distance'] = var_total
    info_u['Uniformity rate of lead distance'] = U_lead



    #### 2.2 The average stop and goes within control
    index_lead_vel = find_topic('v_ref', topics_name_s)
    lead_vel = np.array(data_s[index_lead_vel[0]])  # [x, 7]
    lead_vel = lead_vel[:, [0, 2]]
    i = 0
    for clip in clips:
        time_start = lead_dis[clip[0], 0]
        time_stop = lead_dis[clip[1], 0]

        start_index = np.argmin(abs(np.ones([lead_vel.shape[0]]) * time_start - lead_vel[:, 0]))
        stop_index = np.argmin(abs(np.ones([lead_vel.shape[0]]) * time_stop - lead_vel[:, 0]))

        # index = 1
        lead_vel_ = lead_vel[start_index:stop_index, :]
        if i == 0:
            lead_vel_new = lead_vel_
        else:
            lead_vel_new = np.concatenate([lead_vel_new, lead_vel_], axis = 0)
        i = i + 1
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(lead_vel_new[:, 0], lead_vel_new[:, 1])
    ax.set_xlabel('time')
    ax.set_ylabel('lead_vel reassemble')
    ax.set_title('lead_vel reassemble')
    ax.legend()
    path = './Result/' + 'lead_vel_reassemble' + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)

    # from scipy.signal import savgol_filter
    # lead_vel[:, 1] = savgol_filter(lead_vel[:, 1], 99, 1 , mode='nearest')

    # lead_vel_s = lead_vel
    # lead_vel_s[:, 1] = smooth_data(lead_vel[:, 1])
    #
    # lead_vel = lead_vel_s

    # calculate the accel
    delta_vel = (lead_vel[1:, 1] - lead_vel[0: -1, 1])
    delta_t = (lead_vel[1:, 0] - lead_vel[0: -1, 0])
    delta_vel = delta_vel / delta_t

    # calculate the predicted lead diatnce
    lead_accel = np.zeros(lead_vel.shape)
    lead_accel[:, 0] = lead_vel[:, 0]
    lead_accel[0: -1, 1] = delta_vel

    # from scipy.signal import savgol_filter
    # lead_accel[:, 1] = savgol_filter(lead_accel[:, 1], 99, 1, mode= 'nearest')

    # for i in range(lead_accel.shape[0]):
    #     if i == 0:
    #         continue
    #     else:
    #         if abs(lead_accel[i, 1] - lead_accel[i - 1, 1]) > 30:
    #             lead_accel[i, 1] = lead_accel[i - 1, 1]
    # index_lead = np.where(lead_accel[:, 1] > 30)[0]
    # lead_accel[index_lead, 1] = 30
    # index_lead = np.where(lead_accel[:, 1] < -30)[0]
    # lead_accel[index_lead, 1] = -30
    i = 0
    for clip in clips:
        time_start = lead_dis[clip[0], 0]
        time_stop = lead_dis[clip[1], 0]

        start_index = np.argmin(abs(np.ones([lead_accel.shape[0]]) * time_start - lead_accel[:, 0]))
        stop_index = np.argmin(abs(np.ones([lead_accel.shape[0]]) * time_stop - lead_accel[:, 0]))
        lead_accel_ = lead_accel[start_index:stop_index, :]
        if i == 0:
            lead_accel_new = lead_accel_
        else:
            lead_accel_new = np.concatenate([lead_accel_new, lead_accel_], axis = 0)
        i = i + 1
    index_lead = np.where(lead_accel_new[:, 1] > 100)[0]
    lead_accel_new[index_lead, 1] = 100
    index_lead = np.where(lead_accel_new[:, 1] < -100)[0]
    lead_accel_new  [index_lead, 1] = -100
    lead_accel_new[:, 1] = smooth_data(lead_accel_new[:,1], n = 200)
    fig, ax = plt.subplots(figsize=[15, 4])
    ax.plot(lead_accel_new[:, 0], lead_accel_new[:, 1] )
    ax.set_xlabel('time')
    ax.set_ylabel('lead_accel reassemble')
    ax.set_title('lead_accel reassemble')
    ax.legend()
    path = './Result/' + 'lead_accel_reassemble' + '.png'
    plt.savefig(path)
    print('Result is saved in ' + path)
    #





    index_vel = find_topic('cmd_vel', topics_name_s)
    cmd_vel = np.array(data_s[index_vel[0]])



    name = 'stops and goes for ego car'
    sg_ego, sg_p_ego, s_sg_ego = S_SG(cmd_vel, allowed_control, sg_velocity_thresh, sg_time_thresh_min, sg_time_thresh_max, name)
    name = 'stops and goes for lead car'
    sg_lead, sg_p_lead, s_sg_lead = S_SG(lead_vel, allowed_control, sg_velocity_thresh, sg_time_thresh_min,
                                         sg_time_thresh_max, name)
    s_sg = sg_p_ego / (sg_p_lead + 1)

    info_sg = {}
    info_sg['stops and goes of the ego car within control time (times / s)'] = sg_p_ego
    info_sg['stops and goes of the ego car within control time (times / s)'] = sg_p_ego
    info_sg['stops and goes of the lead car within control time (times / s)'] = sg_p_lead
    info_sg['S_SG'] = s_sg
    info['stablity of the stops and goes within the control'] = info_sg


    #### 2.3 Velocity stability within the control analysis
    name = 'velocity stability within control of the ego car'
    # var_vel_ego, var_vel_p_ego, S_vel_ego = Var_vel(cmd_vel, allowed_control, name)
    var_vel_ego = Var_vel(cmd_vel, allowed_control, name)
    name = 'velocity stability within control of the lead car'
    # var_vel_lead, var_vel_p_lead, S_vel_lead = Var_vel(lead_vel, allowed_control, name)
    var_vel_lead = Var_vel(lead_vel, allowed_control, name)
    S_vel = var_vel_ego / var_vel_lead



    info_vel = {}
    info_vel['variance of the command velocity within the control time of the ego car'] = var_vel_ego
    info_vel['variance of the velocity within the control time of the lead car'] = var_vel_lead
    info_vel['S_vel'] = S_vel
    info['stablity of the velocity within the control'] = info_vel

    #### 2.4 Acceleration stability within the control analysis
    index_lead = np.where(lead_accel[:, 1] > 100)[0]
    lead_accel[index_lead, 1] = 100
    index_lead = np.where(lead_accel[:, 1] < -100 )[0]
    lead_accel[index_lead, 1] = -100

    index_accel = find_topic('cmd_accel', topics_name_s)
    cmd_accel = np.array(data_s[index_accel[0]])

    i = 0
    for clip in clips:
        time_start = lead_dis[clip[0], 0]
        time_stop = lead_dis[clip[1], 0]

        start_index = np.argmin(abs(np.ones([cmd_accel.shape[0]]) * time_start - cmd_accel[:, 0]))
        stop_index = np.argmin(abs(np.ones([cmd_accel.shape[0]]) * time_stop - cmd_accel[:, 0]))

        cmd_accel_ = cmd_accel[start_index:stop_index, :]
        if i == 0:
            cmd_accel_new = cmd_accel_
        else:
            cmd_accel_new = np.concatenate([cmd_accel_new, cmd_accel_], axis=0)
        i = i + 1

    name = 'acceleration stability within control of the ego car'
    var_accel_ego = Var_accel(cmd_accel_new, allowed_control, name)
    # var_accel_ego, var_accel_p_ego, S_accel_ego = Var_accel(cmd_accel_new, allowed_control, name)
    name = 'acceleration stability within control of the lead car'
    # var_accel_lead, var_accel_p_lead, S_accel_lead = Var_accel(lead_accel_new, allowed_control,name)
    var_accel_lead = Var_accel(lead_accel_new, allowed_control, name)

    S_accel = var_accel_ego/var_accel_lead

    # plot_compared(cmd_accel_new, lead_accel_new, length=cmd_accel_new.shape[0])
    plot_compared(cmd_vel, lead_vel, length=cmd_vel.shape[0])

    info_accel = {}
    info_accel['variance of the command acceleration within the control time of the ego car'] = var_accel_ego
    info_accel['variance of the acceleration within the control time of the lead car'] = var_accel_lead
    info_accel['S_accel'] = S_accel
    info['stablity of the acceleration within the control'] = info_accel

    #### 2.5 Uniform score and the stability score of the controller
    u_score, s_score, score = US_controller(U_lead, s_sg, S_vel, S_accel)
    info_score = {}
    info_score['Uniformity score of the controller'] = u_score
    info_score['Stability score of the controller'] = s_score
    info_score['Score of the controller'] = score
    info['Unifomity and stability analysis of the controller'] = info_score

    save_dir = './Result/'
    json_str = json.dumps(info, indent=4)
    path_info = os.path.join(save_dir, 'output.json')
    with open(path_info, 'w') as json_file:
        json_file.write(json_str)

















if __name__ == '__main__':
    print('*' * 120)
    print('Evaluation of the controller analysis Starts')
    main()
    print('Evaluation of the controller analysis Ends')
    print('*' * 120)












