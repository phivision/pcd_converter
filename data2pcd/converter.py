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
import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
import rembg.bg as bg
from .bg_remover import remove_rgbd_bg
import io
from pathlib import Path
from PIL import Image

# global variables
DEFAULT_IMG_ASPECT_RATIO = 4.0 / 3.0
DEFAULT_CAM_POS = (0.0, 0.0, 0.0)
DEFAULT_CAM_EULER_ANG = (0.0, 0.0, 0.0)
# iOS lidar max range
TARGET_MAX_DEPTH = 5.0


class Converter:
    """A tool to convert raw data to PCD format and contains intermediate data for analyze"""
    # Threshold to separate human from background
    MASK_THRESHOLD = 196

    def __init__(self, cam_fov: float = 58.986):
        """Initialize attributes and camera parameters
        The camera parameter reference:
        https://developer.apple.com/library/archive/documentation/DeviceInformation/Reference/iOSDeviceCompatibility/Cameras/Cameras.html

        Args:
            cam_fov: Camera field of view in degree
        """
        self.depth_map = None
        self.map_row = 0
        self.map_col = 0
        self.confidence_map = None
        self.feature_points = None
        self.point_cloud = None
        self.mask = None
        self._max_depth = None
        self._min_depth = None
        self._x = None
        self._y = None
        self._z = None
        self.image = None
        self.image_float = None
        self._container = None
        self._cam_fov_deg = cam_fov
        self._cam_pos = DEFAULT_CAM_POS
        self._cam_euler_ang_deg = DEFAULT_CAM_EULER_ANG
        self._img_aspect_ratio = DEFAULT_IMG_ASPECT_RATIO

    def base64_decoder(self, key: str) -> bytes:
        """Decode base64 data in JSON dict

        Args:
            key: key to access base64 data

        Returns:
            binary data
        """
        return base64.decodebytes(self._container[key].encode('utf-8'))

    def generate_point_cloud(self, background=True):
        """Generate point cloud from depth map

        Args:
            background: if true, keep background, if false, remove background

        """
        if self.depth_map is not None:
            if background:
                output_data = self.depth_map
            else:
                output_data = self.depth_map.copy()
                output_data[self.mask < self.MASK_THRESHOLD] = self.depth_map.max() * 2
            self._x = np.zeros((self.map_row, self.map_col), dtype=np.float32)
            self._y = np.zeros((self.map_row, self.map_col), dtype=np.float32)
            self._z = np.zeros((self.map_row, self.map_col), dtype=np.float32)
            # TODO: this is only simplified algorithm which assume the Z euler angle is 0
            azimuth_deg = (np.arange(self.map_col) - self.map_col / 2.0) * self._cam_fov_deg / \
                self.map_col + self._cam_euler_ang_deg[0]
            azimuth_rad = np.deg2rad(azimuth_deg)
            self._x -= output_data * np.tan(azimuth_rad) + self._cam_pos[0]
            self._z -= output_data + self._cam_pos[1]
            altitude_deg = (np.arange(self.map_row) - self.map_row / 2.0) * self._cam_fov_deg / \
                self._img_aspect_ratio / self.map_row + self._cam_euler_ang_deg[1]
            altitude_rad = np.deg2rad(altitude_deg)
            self._y += output_data * np.tan(altitude_rad.reshape(-1, 1)) + self._cam_pos[2]
            self.point_cloud = np.hstack((np.ravel(self._x).reshape(-1, 1),
                                          np.ravel(self._y).reshape(-1, 1),
                                          np.ravel(self._z).reshape(-1, 1)))
        else:
            raise ValueError("Depth map data is not ready, cannot process!")

    def generate_mask(self, method='u2net'):
        """Use U2NET to generate mask for background removal"""
        if method == 'u2net':
            model = bg.get_model("u2net")
            mask = bg.detect.predict(model, self.image_float).convert("L").resize((self.image_float.shape[1],
                                                                                   self.image_float.shape[0]),
                                                                                  Image.LANCZOS)
            self.mask = np.asarray(mask)
        elif method == 'custom':
            self.mask = remove_rgbd_bg(np.array(self.image.convert("RGB")), self.depth_map)
        else:
            raise(ValueError('Do not support this type of method'))

    def load_json(self, input_file: Path) -> None:
        """Convert JSON input file to PCD output file

        Args:
            input_file: input path of JSON file

        Returns:

        """
        with input_file.open(mode='r') as json_file:
            self._container = json.load(json_file)
            if "pointCloud" in self._container:
                self.feature_points = np.array(self._container["pointCloud"], dtype=np.float32)
            self._max_depth = self._container["maxDepth"]
            self._min_depth = self._container["minDepth"]
            base64_keys = ["depthMapData", "confidenceMapData", "capturedImageData"]
            map_list = ["depth_map", "confidence_map", "image"]
            for key, img in zip(base64_keys, map_list):
                binary_data = self.base64_decoder(key)
                setattr(self, img, Image.open(io.BytesIO(binary_data)))
            self.depth_map = np.array(self.depth_map)[:, :, 0] \
                / 255.0 * (self._max_depth - self._min_depth) + self._min_depth
            self.map_row = self.depth_map.shape[0]
            self.map_col = self.depth_map.shape[1]
            self._img_aspect_ratio = self.map_col / self.map_row
            self.confidence_map = np.array(self.confidence_map)[:, :, 0]
            self.image.thumbnail((self.map_col, self.map_row), Image.ANTIALIAS)
            self.image_float = np.array(self.image) / 255.0

    def export_depth(self, output_file: Path, background=True, method='u2net'):
        """Export depth as Numpy binary file

        Args:
            output_file: Path
            background: if true, keep background, if false, remove background
            method: method to remove background

        """
        with output_file.open(mode='w') as depth_file:
            # remove human background and export depth of only human area
            if not background:
                bg_removed = self.depth_map.copy()
                self.generate_mask(method=method)
                bg_removed[self.mask < self.MASK_THRESHOLD] = TARGET_MAX_DEPTH
                np.save(depth_file.name, bg_removed)
            else:
                np.save(depth_file.name, self.depth_map)

    def export_pcd(self, output_file: Path, background=True):
        """Export point clouds as PCD format

        Args:
            output_file: output path of PCD file
            background: if true, keep background, if false, remove background

        Returns:

        """
        # generate point cloud from depth image
        self.generate_point_cloud(background=background)
        if self.feature_points is not None:
            # PCL only take float32 but by default, numpy is using double
            # here, explicitly using float32 is necessary
            pcd_feature_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.feature_points))
            pcd_cloud_points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.point_cloud))
            pcd_cloud_points.colors = o3d.utility.Vector3dVector(
                self.image_float.reshape(self.map_row * self.map_col, 4)[:, :3])
            file_name = output_file.name
            feature_file = f"feature_{file_name}"
            cloud_file = f"cloud_{file_name}"
            o3d.io.write_point_cloud(f"{output_file.parent}/{feature_file}", pcd_feature_points)
            o3d.io.write_point_cloud(f"{output_file.parent}/{cloud_file}", pcd_cloud_points)

    def visualize(self):
        """Visualize intermediate map and point cloud for analyzsis and debugging"""
        # for debugging only
        if self.image is not None:
            plt.imshow(self.depth_map)
            print(f"The size of depth map: {self.depth_map.shape}")
            plt.show()
            plt.imshow(self.confidence_map)
            print(f"The size of confidence map: {self.confidence_map.shape}")
            plt.show()
            plt.imshow(self.image)
            print(f"The size of rgb image: {self.image.shape}")
            plt.show()
            plt.imshow(self.mask)
            print(f"The size of background mask: {self.mask.shape}")
            plt.show()
