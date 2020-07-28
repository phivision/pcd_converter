# PCD Converter Tool
Convert output JSON files from iOS data logger or other compatible formats to PCL compatible PCD format.

This tool has been tested under Ubuntu 18.04, Python 3.6 with Anaconda Environment.

## Installation
* Create a conda env -> `conda create -n pcd_converter python=3.6`
* Activate conda env -> `conda activate pcd_converter`
* Update conda -> `conda activate pcd_converter`
* Add conda-forge channels -> `conda config --add channels conda-forge`
* Install customized Python-pcl module -> `conda install pcl python-pcl -c artemuk -c conda-forge`
* Install other dependency modules -> `conda install click matplotlib`

## How to use this tool
* After activation of corresponding conda env, run following command in root directory of this project, for example,
if you are using '~/path/input_file.json' as input and 'output_file.pcd' as output.
```shell script
python convert.py --json ~/path/input_file.json --pcd output_file.pcd
```

This tool is developed by Fanghao Yang, Phi Vision Inc. 2020