# PCD Converter Tool
Convert output JSON files from iOS data logger or other compatible formats to PCL compatible PCD format.

This tool has been tested under Ubuntu 18.04, Python 3.6 with Anaconda Environment.

## Installation
* Create a conda env -> `conda create -n pcd_converter python=3.6`
* Activate conda env -> `conda activate pcd_converter`
* Add conda-forge channels -> `conda config --add channels conda-forge`
* Install Open3D library to get richer 3D processing functions -> `conda install -c open3d-admin open3d`
* Install pillow for image processing -> `conda install pillow`
* Install scikit learn -> `conda install -c intel scikit-learn`
* Install Pandas, plyfile, tqdm -> `conda install pandas plyfile tqdm`
* Install other dependency modules -> `conda install addict pyyaml click matplotlib`

## How to use this tool
* After activation of corresponding conda env, run following command in root directory of this project, for example,
if you are using '~/path/input_file.json' as input and 'output_file.pcd' as output.
```shell script
python convert.py --json ~/path/input_file.json --pcd output_file.pcd
```

This tool is developed by Fanghao Yang, Phi Vision Inc. 2020