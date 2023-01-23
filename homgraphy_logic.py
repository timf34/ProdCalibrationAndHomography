import cv2
import os
import numpy as np

from typing import NamedTuple, Tuple, List, Dict

from config import CameraJetson3, CameraJetson1, RealWorldPitchCoords, HOMOGRAPHY_SAVE_DIR
from utils import save_np_matrix


class HomographyLogic:
    def __init__(self):
        pass


def compute_homographies(file_name: str = "new_h_jetson") -> Dict[str, np.ndarray]:
    jetson1 = CameraJetson1()
    jetson3 = CameraJetson3()
    real_world = RealWorldPitchCoords()
    real_world.convert_to_pixel_coords()
    print(real_world.__dict__)

    # Create an empty np array to store the pixel coordinates
    j1_arr = np.array([])
    j2_arr = np.array([])


    # Get the pixel coordinates from the jetson cameras, and the real world coordinates. Convert to NP.arrays
    world_points1 = np.array([])
    for key in jetson1.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson1.__dict__[key] is not None:
            j1_arr = np.append(j1_arr, jetson1.__dict__[key], axis=0)
            world_points1 = np.append(world_points1, real_world.__dict__[key])

    j1_arr = j1_arr.reshape(-1, 2)
    world_points1 = world_points1.reshape(-1, 2)

    world_points2 = np.array([])
    for key in jetson3.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson3.__dict__[key] is not None:
            j2_arr = np.append(j2_arr, jetson3.__dict__[key], axis=0)
            world_points2 = np.append(world_points2, real_world.__dict__[key])
    j2_arr = j2_arr.reshape(-1, 2)
    world_points2 = world_points2.reshape(-1, 2)

    assert len(j2_arr) == len(world_points2), "The number of points in the jetson 3 array and the real world array are not the same"
    assert len(j1_arr) == len(world_points1), "The number of points in the jetson 1 array and the real world array are not the same"
    assert j1_arr.shape[1] == 2, "The jetson 1 array is not 2D"
    assert j2_arr.shape[1] == 2, "The jetson 2 array is not 2D"

    # Compute the homography
    h1, status1 = cv2.findHomography(j1_arr, world_points1)
    h2, status2 = cv2.findHomography(j2_arr, world_points2)

    # Save the homography
    save_np_matrix(matrix=h1, filename=f"{file_name}1")
    save_np_matrix(matrix=h2, filename=f"{file_name}3")

    # Load the homography
    # h1 = np.load(os.path.join(HOMOGRAPHY_SAVE_DIR, "homography_matrix_jetson1.npy"))
    # h2 = np.load(os.path.join(HOMOGRAPHY_SAVE_DIR, "homography_matrix_jetson3.npy"))

    # Test the homography
    # Fix here: https://answers.opencv.org/question/252/cv2perspectivetransform-with-python/
    # test_point = np.array([[1408, 310], 1.0], dtype='float32')

    # # Transform the point
    test_point = np.array([[1408], [310], [1.0]], dtype='float32')

    transformed_point = h2 @ test_point  # This isn't super accurate but we'll use it just to move on for now.
    transformed_point = transformed_point / transformed_point[2]
    # print(transformed_point)

    # Test with h1
    test_point = np.array([[1062], [817], [1.0]], dtype='float32')
    transformed_point = h1 @ test_point  # This isn't super accurate but we'll use it just to move on for now.
    transformed_point = transformed_point / transformed_point[2]
    # print(transformed_point)

    assert transformed_point.shape == (3, 1), "The transformed point is not 3x1"

    return {"1": h1, "3": h2}


def homography_idx(camera_id):
    # This returns the homography matrix, given a camera number (where the number is a string!)
    homography_dict = compute_homographies()

    if camera_id in homography_dict:
        return homography_dict[camera_id]
    else:
        print(str(camera_id) + ' does not have a key in homography_dict')
        raise KeyError


def inspect_homography():
    # Current homographies
    h_dict = compute_homographies()

    # Legacy homographies
    old_h_dict = {"1": np.load(os.path.join(HOMOGRAPHY_SAVE_DIR, "homography_matrix_jetson1.npy")),
                  "3" : np.load(os.path.join(HOMOGRAPHY_SAVE_DIR, "homography_matrix_jetson3.npy"))}

    # Check the shapes of the homography matrices
    for key, value in h_dict.items():
        print(f"Camera {key} homography matrix shape: {value.shape}")

    # Check the determinant
    for key, value in h_dict.items():
        print(f"Camera {key} homography matrix determinant: {np.linalg.det(value)}")

    # Check if h_dict == old_h_dict
    for key, value in h_dict.items():
        print(f"Camera {key} homography matrix == old homography matrix: {np.array_equal(value, old_h_dict[key])}")
        assert np.array_equal(value, old_h_dict[key]), "The homography matrices are not equal"


def main():
    inspect_homography()


if __name__ == '__main__':
    main()
