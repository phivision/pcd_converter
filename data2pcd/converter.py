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

import base64
import pcl
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
from pathlib import Path


class Converter:
    """A tool to convert raw data to PCD format and contains intermediate data for analyze"""
    def __init__(self):
        self.depth_map = None
        self.confidence_map = None
        self.feature_points = None
        self.point_cloud = None
        self.image = None
        self._container = None

    def base64_decoder(self, key: str) -> bytes:
        """Decode base64 data in JSON dict

        Args:
            key: key to access base64 data

        Returns:
            binary data
        """
        return base64.decodebytes(self._container[key].encode('utf-8'))

    def convert(self, input_file: Path) -> None:
        """Convert JSON input file to PCD output file

        Args:
            input_file: input path of JSON file

        Returns:

        """
        with input_file.open(mode='r') as json_file:
            self._container = json.load(json_file)
            self.feature_points = np.array(self._container["pointCloud"], dtype=np.float32)
            base64_keys = ["depthMapData", "confidenceMapData", "capturedImageData"]
            map_list = ["depth_map", "confidence_map", "image"]
            for key, img in zip(base64_keys, map_list):
                binary_data = self.base64_decoder(key)
                setattr(self, img, mpimg.imread(io.BytesIO(binary_data)))

    def export(self, output_file: Path):
        """Export points as PCD format

        Args:
            output_file: output path of PCD file

        Returns:

        """
        if self.feature_points is not None:
            # PCL only take float32 but by default, numpy is using double
            # here, explicitly using float32 is necessary
            point_cloud = pcl.PointCloud(self.feature_points)
            point_cloud.to_file(str(output_file.absolute()).encode('utf-8'))

    def visualize(self):
        """Visualize intermediate map and point cloud for analyzsis and debugging"""
        # for debugging only
        if self.image is not None:
            plt.imshow(self.depth_map[:, :, 0])
            print(f"The size of depth map: {self.depth_map.shape}")
            plt.show()
            plt.imshow(self.confidence_map[:, :, 0])
            print(f"The size of confidence map: {self.confidence_map.shape}")
            plt.show()
            plt.imshow(self.image)
            print(f"The size of rgb image: {self.image.shape}")
            plt.show()