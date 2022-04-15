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
from tool_algorithm import *
from utils import *

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--bagfile",
                        help="the dir of the bagfile",
                        default='')

    parser.add_argument("--parameter_file",
                        help="the dir of parameter file",
                        default='')

    parser.add_argument("--save_dir",
                        help="the dir to save the outout",
                        default='./split_result')

    parser.add_argument("--outbag_name",
                        help="the dir to save the outout",
                        default='output.bag')

    parser.add_argument("--start_time",
                        type=int,
                        help="the start time to split")

    parser.add_argument("--split_duration",
                        help="the split duration (seconds)",
                        type=int)

    return parser.parse_args()



def main():
    info = {}
    args = parse_args()
    inbag = args.bagfile
    start_time = args.start_time
    duration = args.split_duration
    save_dir = args.save_dir
    outbag = os.path.join(args.save_dir, args.outbag_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ## save the parameter of the input bagfile
    parameter_file_dir = args.parameter_file
    parameter_file = pd.read_csv(parameter_file_dir).to_dict('records')

    info['parameter configuration of the splitted bagfile'] = parameter_file[0]

    ## read the bagfile
    print('Read the bagfile from: ' + inbag)
    topics_num, topics_name, bag = read_bagfiles_and_topics(inbag)
    data = transfer_format(bag, topics_name, topics_num)
    info_inbag = bagfile_info(data, topics_name, topics_num)
    info['metadata of the input bagfile'] = info_inbag

    time_info = time_analysis(data, topics_name, topics_num, start_time, duration)

    ## split the bagfile
    splitter = RosBagSplitter(inbag=inbag, outdir=save_dir, outfile=outbag, topics_name=topics_name,
                              time_info=time_info)
    splitter.split()

    ## save the parameter of the output bagfile
    print('Read the output bagfile from: ' + outbag)
    topics_num, topics_name, bag = read_bagfiles_and_topics(outbag)
    data = transfer_format(bag, topics_name, topics_num)
    info_outbag = bagfile_info(data, topics_name, topics_num)
    info['metadata of the output bagfile'] = info_outbag

    #### save as json file
    json_str = json.dumps(info, indent=4)
    path_info = os.path.join(save_dir, 'output_info.json')
    with open(path_info, 'w') as json_file:
        json_file.write(json_str)

if __name__=='__main__':
    print('*' * 120)
    print('Bagfile Split Starts!')
    main()
    print('Bagfile Split Finished!')
    print('*' * 120)
