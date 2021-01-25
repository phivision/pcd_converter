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
from data2pcd.bg_remover import learning_bg
import click
from copy import copy
from pathlib import Path


@click.command()
@click.option('--json', help="The path of input JSON file")
@click.option('--pcd', help="The path of output PCD file")
@click.option('--depth', help="The path of output depth npy file")
@click.option('--use_bg', type=bool, default=True, help="If keep the background")
@click.option('--bg_path', help="The path to background maps")
@click.option('--method', default='u2net', help="method to remove background, "
                                                "including 'u2net', 'dynamic' and 'static'")
def convert(json, pcd, depth, use_bg, bg_path, method):
    """Convert JSON to PCD file from command line"""
    json_converter = json2pcd.Converter()
    json_converter.load_json(Path(json))
    if pcd:
        json_converter.export_pcd(Path(pcd), use_bg=use_bg)
        print(f"Successfully converted point cloud data from {json} to {pcd}")
    if depth:
        if bg_path and method == 'static':
            bg_files = Path(bg_path).glob("*.json")
            bg_converter = json2pcd.Converter()
            bg_list = []
            for bg_file in bg_files:
                bg_converter.load_json(bg_file)
                bg_list.append(copy(bg_converter.depth_map))
            bg, bg_std = learning_bg(bg_list)
            json_converter.set_background(bg, bg_std)
            print("Static background is loaded to the converter!")
        json_converter.export_depth(Path(depth), use_bg=use_bg, method=method)
        print(f"Successfully converted depth data from {json} to {depth}")


if __name__ == "__main__":
    # execute only if run as a script
    convert()
