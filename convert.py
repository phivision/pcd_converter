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
Individual file converter for JSON

By Fanghao Yang, 07/27/2020
"""

import data2pcd.converter as json2pcd
import click
from pathlib import Path


@click.command()
@click.option('--json', help="The path of input JSON file")
@click.option('--pcd', help="The path of output PCD file")
@click.option('--depth', help="The path of output depth npy file")
def convert(json, pcd, depth):
    """Convert JSON to PCD file from command line"""
    json_converter = json2pcd.Converter()
    json_converter.load_json(Path(json))
    if pcd:
        json_converter.export_pcd(Path(pcd))
        print(f"Successfully converted point cloud data from {json} to {pcd}")
    if depth:
        json_converter.export_depth(Path(depth))
        print(f"Successfully converted depth data from {json} to {pcd}")


if __name__ == "__main__":
    # execute only if run as a script
    convert()
