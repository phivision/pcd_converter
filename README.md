# PCD Converter Tool
Convert output JSON files from iOS data logger or other compatible formats to 
PCL compatible PCD format or Numpy compatible depth array.

And visualization tool is provided for quick checkup.

This tool has been tested under Ubuntu 18.04 and MacOS 11, Python 3.8 with Anaconda Environment.

## Installation
* Create a conda env -> `conda create -n pcd_converter python=3.8`
* Activate conda env -> `conda activate pcd_converter`
* Install background remover -> `pip install rembg`
* Add conda-forge channels -> `conda config --add channels conda-forge`
* Install Open3D library to get richer 3D processing functions -> `conda install -c open3d-admin open3d`
* Install scikit learn -> `conda install -c intel scikit-learn`
* Install other dependency modules -> `conda install pandas plyfile addict pyyaml click`

## How to use this tool
* After activation of corresponding conda env, run following command in root directory of this project, for example,
if you are using '~/path/input_file.json' as input and 'output_file.pcd' as output.
```shell script
python convert.py --json ~/path/input_file.json --pcd output_file.pcd
```
You may only visualize the JSON data without conversion
```shell script
python plot.py --json ~/path/input_file.json
```
This tool is developed by Fanghao Yang, Phi Vision Inc. 2020