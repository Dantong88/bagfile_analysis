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
from tool1_algorithm import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagfile1",
                        help="the dir of bagfile 1",
                        default='../bagfile/0804/2021_08_04_00_05_20_2T3H1RFV8LC057037following_real_vehicle_rl0719_enable_true.bag')

    parser.add_argument("--bagfile2",
                        help="the dir of bagfile 2",
                        default='../bagfile/0804/2021_08_04_00_08_00_2T3H1RFV8LC057037following_real_vehicle_rl0719_enable_true.bag')

    parser.add_argument("--parameter_file1",
                        help="the dir of parameter file 1",
                        default='../bagfile/0804/2021_08_04_00_05_18_rosparams_following_real_vehicle_rl0719_enable_true.csv')

    parser.add_argument("--parameter_file2",
                        help="the dir of parameter file 2",
                        default='../bagfile/0804/2021_08_04_00_07_59_rosparams_following_real_vehicle_rl0719_enable_true.csv')

    parser.add_argument("--save_dir",
                        help="the dir to save the outout",
                        default='./output')

    return parser.parse_args()




def main():
    info = {}
    args = parse_args()
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## save the parameter info of the bagfile
    info1 = {}
    parameter_file_dir1 = args.parameter_file1
    parameter_file_dir2 = args.parameter_file2

    parameter_file1 = pd.read_csv(parameter_file_dir1).to_dict('records')
    parameter_file2 = pd.read_csv(parameter_file_dir2).to_dict('records')
    info1['bagfile 1'] = parameter_file1[0]
    info1['bagfile 2'] = parameter_file2[0]

    info['parameter configuration of the bagfiles'] = info1

    ## read the bagfile
    bagfile1_path = args.bagfile1
    print('Read the first bagfile from: ' + bagfile1_path)
    topics_num_1,topics_name_1, bag_1 = read_bagfiles_and_topics(bagfile1_path)

    ## read the bagfile
    bagfile2_path = args.bagfile2
    print('Read the second bagfile from: ' + bagfile2_path)
    topics_num_2, topics_name_2, bag_2 = read_bagfiles_and_topics(bagfile2_path)

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

    data_1 = transfer_format(bag_1, topics_name_1, topics_num_1)
    data_2 = transfer_format(bag_2, topics_name_2, topics_num_2)

    #### 1. find the maximum and minimum of the command acceleration
    info2 = {}
    max_cmd_accel1, max_cmd_accel2, min_cmd_accel1, min_cmd_accel2, max_time1, max_time2, min_time1, min_time2,\
        l1_1, l2_1, inf_1, std1, l1_2, l2_2, inf_2, std2= cmd_accel_analysis(data_1, data_2, topics_name_1, topics_name_2, args)
    max_cmd = {}
    max_cmd['bagfile 1'] = max_cmd_accel1
    max_cmd['time for maximum accel (bagfile 1)'] = max_time1.tolist()
    max_cmd['bagfile 2'] = max_cmd_accel2
    max_cmd['time for maximum accel (bagfile 2)'] = max_time2.tolist()

    min_cmd = {}
    min_cmd['bagfile 1'] = min_cmd_accel1
    min_cmd['time for minimum accel (bagfile 1)'] = min_time1.tolist()
    min_cmd['bagfile 2'] = min_cmd_accel2
    min_cmd['time for minimum accel (bagfile 2)'] = min_time2.tolist()

    L1_norm = {}
    L1_norm['bagfile 1'] = l1_1
    L1_norm['bagfile 2'] = l1_2

    L2_norm = {}
    L2_norm['bagfile 1'] = l2_1
    L2_norm['bagfile 2'] = l2_2

    inf_norm = {}
    inf_norm['bagfile 1'] = inf_1
    inf_norm['bagfile 2'] = inf_2

    std = {}
    std['bagfile 1'] = std1
    std['bagfile 2'] = std2

    info2['maximum acceleration (m/s^2)'] = max_cmd
    info2['minimum acceleration (m/s^2)'] = min_cmd
    info2['L1 norm of the cmd_accel'] = L1_norm
    info2['L2 norm of the cmd_accel'] = L2_norm
    info2['L_infty norm of the cmd_accel'] = inf_norm
    info2['Std of the cmd_accel'] = std

    info['quantitative analysis of command acceleration of each bagfiles'] = info2


    #### 2. find the maximum and minimum of the v_ref
    info3 = {}
    max_v_ref1, max_v_ref2, min_v_ref1, min_v_ref2, max_time1, max_time2, min_time1, min_time2,\
        l1_1, l2_1, inf_1, std1, l1_2, l2_2, inf_2, std2= v_ref_analysis(data_1, data_2, topics_name_1,
                                                                                    topics_name_2, args)
    max_v_ref = {}
    max_v_ref['bagfile 1'] = max_v_ref1
    max_v_ref['time for maximum v_ref (bagfile 1)'] = max_time1.tolist()
    max_v_ref['bagfile 2'] = max_v_ref2
    max_v_ref['time for maximum v_ref (bagfile 2)'] = max_time2.tolist()

    min_v_ref = {}
    min_v_ref['bagfile 1'] = min_v_ref1
    min_v_ref['time for minimum v_ref (bagfile 1)'] = min_time1.tolist()
    min_v_ref['bagfile 2'] = min_v_ref2
    min_v_ref['time for minimum v_ref (bagfile 2)'] = min_time2.tolist()

    L1_norm = {}
    L1_norm['bagfile 1'] = l1_1
    L1_norm['bagfile 2'] = l1_2

    L2_norm = {}
    L2_norm['bagfile 1'] = l2_1
    L2_norm['bagfile 2'] = l2_2

    inf_norm = {}
    inf_norm['bagfile 1'] = inf_1
    inf_norm['bagfile 2'] = inf_2

    std = {}
    std['bagfile 1'] = std1
    std['bagfile 2'] = std2

    info3['maximum acceleration (m/s^2)'] = max_v_ref
    info3['minimum acceleration (m/s^2)'] = min_v_ref
    info3['L1 norm of the cmd_accel'] = L1_norm
    info3['L2 norm of the cmd_accel'] = L2_norm
    info3['L_infty norm of the cmd_accel'] = inf_norm
    info3['Std of the cmd_accel'] = std

    info['quantitative analysis of velocity reference of each bagfiles'] = info3


    #### 3. cmd_accel comparison
    info4 = {}

    l1, l2, inf, std, info_ = cmd_accel_comp(data_1, data_2, topics_name_1, topics_name_2, args)

    info4['L1 norm of the difference of the cmd_accel between bagfile 1 and bagfile 2'] = l1
    info4['L2 norm of the difference of the cmd_accel between bagfile 1 and bagfile 2'] = l2
    info4['L_infty norm of the difference of the cmd_accel between bagfile 1 and bagfile 2'] = inf
    info4['standard deviation of the difference of the cmd_accel between bagfile 1 and bagfile 2'] = std

    info4['changes between bagfile 1 and bagfile 2 for cmd_accel'] = info_

    info['quantitative comparison between bagfile 1 and bagfile 2 for command acceleration'] = info4


    #### 4. v_ref comparison
    info5 = {}
    l1, l2, inf, std, info_ = v_ref_comp(data_1, data_2, topics_name_1, topics_name_2, args)

    info5['L1 norm of the difference of the v_ref between bagfile 1 and bagfile 2'] = l1
    info5['L2 norm of the difference of the v_ref between bagfile 1 and bagfile 2'] = l2
    info5['L_infty norm of the difference of the v_ref between bagfile 1 and bagfile 2'] = inf
    info5['standard deviation of the difference of the v_ref between bagfile 1 and bagfile 2'] = std

    info5['changes between bagfile 1 and abgfile 2 for v_ref'] = info_

    info['quantitative comparison between bagfile 1 and bagfile 2 for velocity reference'] = info5



    #### save as json file
    json_str = json.dumps(info, indent=4)
    path_info = os.path.join(save_dir, 'output.json')
    with open(path_info, 'w') as json_file:
        json_file.write(json_str)










if __name__ == '__main__':
    print('*' * 120)
    print('Bagfile analysis Starts')
    main()
    print('Bagfile analysis Finished')
    print('*' * 120)