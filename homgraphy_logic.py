import cv2
import os
import numpy as np

from typing import NamedTuple, Tuple, List, Dict, Union

from config import CameraJetson3, CameraJetson1, RealWorldPitchCoords, HOMOGRAPHY_SAVE_DIR
from utils import save_np_matrix

# Note: fix here for some prev error: https://answers.opencv.org/question/252/cv2perspectivetransform-with-python/


class HomographyLogic:
    def __init__(self, convert_to_pixels: bool = False):
        self.jetson1 = CameraJetson1()
        self.jetson3 = CameraJetson3()
        self.real_world_pitch_coords = RealWorldPitchCoords()
        if convert_to_pixels:
            self.real_world_pitch_coords.convert_to_pixel_coords()  # Instead of to metres
        self.homographies: Dict[str, np.ndarray] = self.get_homographies()

    def get_real_world_coords(self, jetson_camera: Union[CameraJetson3, CameraJetson1]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts jetson camera coords + real world coords to np.ndarray. It also skips None values.
        :param jetson_camera:
        :return:
        """
        # Initialize two empty arrays to store the pixel coords
        camera_coords: np.ndarray = np.array([])
        world_points = np.array([])

        for key in jetson_camera.__dict__.keys():
            if key in self.real_world_pitch_coords.__dict__.keys() and self.real_world_pitch_coords.__dict__[key] is not \
                    None and jetson_camera.__dict__[key] is not None:
                camera_coords = np.append(camera_coords, jetson_camera.__dict__[key], axis=0)
                world_points = np.append(world_points, self.real_world_pitch_coords.__dict__[key])

        camera_coords = camera_coords.reshape(-1, 2)
        world_points = world_points.reshape(-1, 2)

        assert len(camera_coords) == len(world_points), "The number of points in the camera array doesn't equal the real world array"
        assert camera_coords.shape[1] == 2, "The camera coords array should have 2 columns (be 2D)"

        return camera_coords, world_points

    def find_homography(self, camera_coords: np.ndarray, real_world_coords: np.ndarray) -> np.ndarray:
        """
        Computes the homography matrix for a given camera and real world coords
        """
        h, status = cv2.findHomography(camera_coords, real_world_coords)
        return h

    def get_homographies(self) -> Dict[str, np.ndarray]:
        c1, r1 = self.get_real_world_coords(jetson_camera=self.jetson1)
        c3, r3 = self.get_real_world_coords(jetson_camera=self.jetson3)

        h1 = self.find_homography(camera_coords=c1, real_world_coords=r1)
        h3 = self.find_homography(camera_coords=c3, real_world_coords=r3)

        return {"1": h1, "3": h3}

    @staticmethod
    def perform_homography(homography: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Performs homography - only operates on a single point for now."""
        assert len(point) == 3, "Point should be of length 3"  # Point should be of the form [x, y, 1], not [x, y]
        transformed_point = homography @ point
        return transformed_point / transformed_point[2]


def homography_idx(camera_id):
    # This returns the homography matrix, given a camera number (where the number is a string!)
    h_logic = HomographyLogic()
    homography_dict = h_logic.get_homographies()

    if camera_id in homography_dict:
        return homography_dict[camera_id]
    print(f'{str(camera_id)} does not have a key in homography_dict')
    raise KeyError


def inspect_homography():
    # Current homographies
    h_logic = HomographyLogic()
    h_dict = h_logic.get_homographies()

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
    # inspect_homography()

    # New class
    h_logic = HomographyLogic()
    new_h = h_logic.get_homographies()
    print(type(new_h["1"]))


if __name__ == '__main__':
    main()
