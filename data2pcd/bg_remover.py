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

import face_recognition
import numpy as np
import pylab
from skimage.filters import threshold_multiotsu, gaussian
from skimage.feature import canny
from skimage.measure import label
from skimage.morphology import disk, dilation, erosion

PARTICLE_FILTER_SIZE = 12


def get_face_center(rgb_image: np.ndarray, debug=False):
    """Get a list of center coordinate of faces in the image

    Args:
        rgb_image: rgb image
        debug: if in debug mode

    Returns:
        a list of face center coordinates
    """
    # up-sampling to find smaller face with CNN model
    # HOG model may not detect some face profiles
    face_locations = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=2, model='cnn')
    face_centers = []
    for loc in face_locations:
        face_centers.append(((loc[1] + loc[3]) // 2, (loc[0] + loc[2]) // 2))
    if debug:
        print(f"Bounding boxes of face: {face_locations}")
        pylab.figure()
        pylab.imshow(rgb_image)
        for center in face_centers:
            pylab.plot(center[0], center[1], '+', color='firebrick')
        pylab.show()
    return face_centers


def post_processing_mask(mask, face_center):
    """Remove non-human blobs and small particles in the mask

    Args:
        mask: input mask with first background subtraction
        face_center: coordinate of a face center

    Returns:
        processed mask
    """
    # remove noise particles from mask
    mask = gaussian(mask, sigma=1 / (4.0 * PARTICLE_FILTER_SIZE))
    # component labeling
    blobs = mask > 0.7 * mask.mean()
    blob_labels = label(blobs, background=0, connectivity=1)
    human_label = blob_labels[face_center[1], face_center[0]]
    mask[blob_labels != human_label] = 0
    return mask


def remove_rgbd_bg(rgb_image: np.ndarray,
                   depth_image: np.ndarray,
                   floor_threshold=0.01,
                   debug=False) -> np.ndarray:
    """
    Remove background in RGBD images. This version only works for one person.
    Args:
        rgb_image: input RGB image
        depth_image: input depth image
        floor_threshold: threshold to remove floor
        debug: if debug mode

    Returns:
        mask of foreground (human)
    """
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        raise(IndexError('Input index mismatch!'))

    face_centers = get_face_center(rgb_image, debug=debug)
    # only take first person
    face_depth = depth_image[face_centers[0][1], face_centers[0][0]]
    # multiple otsu's thresholding
    thresholds = np.array(threshold_multiotsu(depth_image))
    if debug:
        print(f"{thresholds} for ostu's methods to separate foreground, background and human")
    assert face_depth > thresholds[0], face_depth < thresholds[1]
    print("successfully separate foreground and background!")
    if debug:
        segments = np.digitize(depth_image, bins=thresholds)
        pylab.imshow(segments, cmap='jet')
        pylab.show()
    mask = np.full(depth_image.shape, 1.0)
    # remove foreground and background from mask
    mask[depth_image < thresholds[0]] = 0
    mask[depth_image > thresholds[1]] = 0
    map_height = depth_image.shape[0]
    map_width = depth_image.shape[1]
    # detect if floor exist in the image
    if np.sum(mask[-2:, :] > 0) > map_width:
        # count how many pixels were kept in last two rows
        # if there is no floor but feet, the total count shall be less than half
        print("detected floors")
        # remove floor
        slope_array = np.median(depth_image[-2, :] + depth_image[-1, :] - depth_image[-3, :] - depth_image[-4, :]) / 4
        mask[-2:, :] = 0
        for i in range(-3, -map_height//2, -1):
            pred_floor_depth = depth_image[-1, :] + (i+1) * slope_array - floor_threshold
            floor_diff = depth_image[i, :] - pred_floor_depth
            if debug:
                print(f"median floor error in row {i}: {np.median(floor_diff)}")
            floor_idx = floor_diff > 0
            mask[i, floor_idx] = 0
    mask = post_processing_mask(mask, face_centers[0])
    if debug:
        pylab.imshow(mask)
        pylab.show()
    return mask


def subtract_bg(rgb_image: np.ndarray,
                depth_image: np.ndarray,
                background: np.ndarray,
                std: np.ndarray,
                threshold=0.0,
                debug=False) -> np.ndarray:
    """Subtract background from the depth map to get mask of moving objects

    Args:
        rgb_image: rgb image
        depth_image: depth image
        background: learned background
        std: standard deviation of background depth
        threshold: to separate object from background, use this threshold with unit meter
        debug: if in debug mode

    Returns:
        mask as 2d uint8 array
    """
    mask = np.full(depth_image.shape, 1.0)
    mask_diff = background - depth_image
    # detect the edge of moving objects
    mask_canny = canny(mask_diff, sigma=3)
    if debug:
        pylab.imshow(mask_diff)
        pylab.show()
        pylab.imshow(mask_canny)
        pylab.show()
    mask[mask_diff < std] = 0.0
    # subtract edges
    mask[mask_canny] = 0.0
    face_centers = get_face_center(rgb_image, debug=debug)
    if len(face_centers) > 0:
        mask = post_processing_mask(mask, face_centers[0])
        # refill the canny edges on mask
        selem = disk(1)
        mask = dilation(mask, selem)
        mask = erosion(mask, selem)
    if debug:
        pylab.imshow(mask)
        pylab.show()
    return mask


def learning_bg(background_images: list) -> tuple:
    """Learning static background from a list of background

    Args:
        background_images: a list of background images

    Returns:
        median value of image stacks and its pixel-wise std
    """
    assert(len(background_images[0].shape) == 2)
    image_stack = np.stack(background_images, axis=2)
    return np.median(image_stack, axis=2), np.std(image_stack, axis=2)
