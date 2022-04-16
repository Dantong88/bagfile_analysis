# Bagpy Analysis [Usage]
## Brief introduction
This repo includes two bagfile analysis tools: bagfile split tool and the bagfile analysis tool.

The bagfile split tool can split the input bagfile based on timestamp, the user can assign the start time and duration of the target bagfile and will output the bagfile clip of the specified time interval.

The bagfile analysis tool can deliver topic-based bagfile data analysis and comparison analysis. The user can input two bagfiles and the corresponding .csv parameter files, then output the analysis as some plots and json file. The details of the analysis topics can refer to my report. *(TODO later)

## Create an environment
```
conda create -n bagpy_analysis python=3.8
source activate bagpy_analysis
```

## Get the repo and prepare the bagfiles
User can get the repo by the following command:
```
git clone https://github.com/Dantong88/bagfile_analysis_tool
cd bagfile_analysis_tool
```
Then, put your bagfile (user can use any bagfiles as input for demo test) and supported .csv parameter file into the folder and save it like:
```
bagpy_analysis_tool/
├── LICENSE
├── README.md
├── split_tool.py
├── analysis_tool.py
├── tool_algorthim.py
├── utils.py
├── your_bagfile.bag
├── bagfile_para.csv
├── requirements.py
```

## Install the requirements
```
pip install -r requirements.txt
```

## Bagfile Split Tool
### 1. Archetecture of the input parameter of the split tool
```
split_tool.py [-h] 
              [--bagfile BAGFILE]                       the dir of the bagfile
              [--parameter_file PARAMETER_FILE]         the dir of parameter file
              [--save_dir SAVE_DIR]                     the dir to save the output
              [--bag_name OUTBAG_NAME]                  the name of the output bagfile
              [--start_time START_TIME]                 the start time to split
              [--split_duration SPLIT_DURATION]         the split duration (seconds)
```
User can also use ```python split_tool.py -h``` to see the help.
### 2. Demo
User can run the demo by the following command:
```
python split_tool.py --bagfile your_bagfile_dir \
                --parameter_file your_parafile_dir \
                --save_dir ./split_result \
                --outbag_name output.bag \
                --start_time 0 \
                --split_duration 30 \
```
For the  ```start_time ```, it should be clarified that the user can either input timestamp as an absolute start time (```e.g.1628031922```), but if user does not know the details of the time, can also just input  ```0```, then the split tool will spilt the bagfile from the start of that file.
User can assign the save dir and the name of splitted bagfile, the start time to split and the duration, the duration is with the unit of second, then it will output the splitted bagfile in the assigned dir with the archetecture below:
```
bagpy_analysis_tool/
├── LICENSE
├── README.md
├── split_result/
│   └── output.bag
│   └── output_info.json
├── split_tool.py
├── analysis_tool.py
├── tool_algorthim.py
├── utils.py
├── your_bagfile.bag
├── bagfile_para.csv
```
The  ```output_info.json``` includes the information of the original bagfile (the topis, dimension and etc.) and the information of the output bagfile (the topis, dimension and etc.)

## Bagfile Analysis Tool
### 1. Archetecture of the input parameter of the analysis tool
```
analysis_tool.py [--bagfile1 BAGFILE1]                      the dir of bagfile 1
                 [--bagfile2 BAGFILE2]                      the dir of bagfile 2
                 [--parameter_file1 PARAMETER_FILE1]        the dir of parameter file 1
                 [--parameter_file2 PARAMETER_FILE2]        the dir of parameter file 2
                 [--save_dir SAVE_DIR]                      the dir to save the outout
```
User can also use ```python analysis_tool.py -h``` to see the help.
### 2. Demo
User can run the demo by the following command:
```
python analysis_tool.py --bagfile1 your_bagfile1 \
                        --bagfile2 your_bagfile2 \                     
                        --parameter_file1 your_parafile1 \
                        --parameter_file2 your_parafile2 \
                        --save_dir your_output_dir
```
User can assign the save dir and the name of splitted bagfile, the start time to split and the duration, the duration is with the unit of second, then it will output the splitted bagfile in the assigned dir with the archetecture below:
```
bagpy_analysis_tool/
├── LICENSE
├── README.md
├── output/
    └── output.json
    ├── v_ref_comparison.png
    ├── v_ref_difference.png
    ├── cmd_accel_comparison.png
    └── cmd_accel_difference.png
├── split_tool.py
├── analysis_tool.py
├── tool_algorthim.py
├── utils.py
├── your_bagfile1.bag
├── your_bagfile2.bag
├── bagfile1_para.csv
├── bagfile2_para.csv
```
The  ```output.json``` includes the information of the original bagfile and the comparison results. It will also output some plots showing the comparison results (details for this comparison can refer to my report).





