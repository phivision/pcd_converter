# Phi Vision, Inc.
# __________________

# [2020] Phi Vision, Inc.  All Rights Reserved.

# NOTICE:  All information contained herein is, and remains
# the property of Adobe Systems Incorporated and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Phi Vision, Inc
# and its suppliers and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
# Dissemination of this information or reproduction of this material
# is strictly forbidden unless prior written permission is obtained
# from Phi Vision, Inc.

"""
Converter from JSON data to PCD data

By Fanghao Yang, 07/27/2020
"""

import pcl
import json
import numpy as np
from pathlib import Path


def converter(input_file: Path, output_file: Path) -> None:
    """

    Args:
        input_file: input path of JSON file
        output_file: output path of PCD file

    Returns:

    """
    with input_file.open(mode='r') as json_file:
        container_dict = json.load(json_file)
        point_cloud_list = container_dict["pointCloud"]
    if point_cloud_list:
        # PCL only take float32 but by default, numpy is using double
        # here, explicitly using float32 is necessary
        point_cloud = pcl.PointCloud(np.array(point_cloud_list, dtype=np.float32))
        point_cloud.to_file(str(output_file.absolute()).encode('utf-8'))
