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
Convert JSON file and plot the data

By Fanghao Yang, 07/27/2020
"""

import data2pcd.converter as json2pcd
import click
import os
from pathlib import Path


@click.command()
@click.option('--json', help="The path of input JSON file")
@click.option('--depth', default=True, type=bool, help="If plot depth map")
@click.option('--image', default=True, type=bool, help="If plot rgb image")
@click.option('--confidence', default=False, type=bool, help="If plot confidence map")
@click.option('--mask', default=False, type=bool, help="If plot mask")
def plot(json, depth, image, confidence, mask):
    """Plot JSON from command line"""
    json_converter = json2pcd.Converter()
    json_path = Path(json)
    if json_path.is_dir():
        json_files = sorted(json_path.glob("*.json"), key=os.path.getmtime)
        for json_file in json_files:
            print(f"Plot json file: {json_file}")
            json_converter.load_json(Path(json_file))
            json_converter.visualize(name=str(json_file.stem),
                                     depth=depth,
                                     image=image,
                                     confidence=confidence,
                                     mask=mask)
    else:
        json_converter.load_json(json_path)
        json_converter.visualize(name=str(json_path.stem),
                                 depth=depth,
                                 image=image,
                                 confidence=confidence,
                                 mask=mask)


if __name__ == "__main__":
    # execute only if run as a script
    plot()
