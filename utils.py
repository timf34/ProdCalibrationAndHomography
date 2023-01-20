import os
import numpy as np

from config import CameraJetson3, CameraJetson1, RealWorldPitchCoords

def save_np_matrix(
        matrix: np.ndarray,
        filename: str,
        directory: str = "data/homography_matrices/"
) -> None:
    """
    Save the homography matrix to a file, first ensuring the directory exists
    :param homography_matrix:
    :param filename:
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory + filename, matrix)


def get_coords_as_array():
    jetson1 = CameraJetson1()
    real_world = RealWorldPitchCoords()

    # Create an empty np array to store the pixel coordinates
    j1_arr = np.array([])

    # Get the pixel coordinates from the jetson cameras, and the real world coordinates. Convert to NP.arrays
    world_points1 = np.array([])
    for key in jetson1.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson1.__dict__[
            key] is not None:
            j1_arr = np.append(j1_arr, jetson1.__dict__[key],
                               axis=0)
            world_points1 = np.append(world_points1, real_world.__dict__[key])

    j1_arr = j1_arr.reshape(-1, 2)
    world_points1 = world_points1.reshape(-1, 2)

    assert len(j1_arr) == len(
        world_points1), f"The number of points in the jetson 1 array and the real world array are not the same: len(" \
                        f"j1_arr) = {len(j1_arr)}, len(world_points1) = {len(world_points1)} "
    assert j1_arr.shape[1] == 2, "The jetson 1 array is not 2D"

    return j1_arr, world_points1


def get_all_coords_as_arrays():
    jetson1 = CameraJetson1()
    jetson3 = CameraJetson3()
    real_world = RealWorldPitchCoords()

    # Create an empty np array to store the pixel coordinates
    j1_arr = np.array([])
    j2_arr = np.array([])

    # Get the pixel coordinates from the jetson cameras, and the real world coordinates. Convert to NP.arrays
    world_points1 = np.array([])
    for key in jetson1.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson1.__dict__[
            key] is not None:
            j1_arr = np.append(j1_arr, jetson1.__dict__[key],
                               axis=0)
            world_points1 = np.append(world_points1, real_world.__dict__[key])

    j1_arr = j1_arr.reshape(-1, 2)
    world_points1 = world_points1.reshape(-1, 2)

    world_points2 = np.array([])
    for key in jetson3.__dict__.keys():
        if key in real_world.__dict__.keys() and real_world.__dict__[key] is not None and jetson3.__dict__[
            key] is not None:
            j2_arr = np.append(j2_arr, jetson3.__dict__[key], axis=0)
            world_points2 = np.append(world_points2, real_world.__dict__[key])
    j2_arr = j2_arr.reshape(-1, 2)
    world_points2 = world_points2.reshape(-1, 2)

    assert len(j2_arr) == len(
        world_points2), "The number of points in the jetson 3 array and the real world array are not the same"
    assert len(j1_arr) == len(
        world_points1), "The number of points in the jetson 1 array and the real world array are not the same"
    assert j1_arr.shape[1] == 2, "The jetson 1 array is not 2D"
    assert j2_arr.shape[1] == 2, "The jetson 2 array is not 2D"

    return j1_arr, j2_arr, world_points1, world_points2