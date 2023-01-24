import cv2
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Iterator

from config import CameraJetson3, CameraJetson1, RealWorldPitchCoords
from homgraphy_logic import HomographyLogic

np.set_printoptions(suppress=True)  # Suppress scientific notation


class VisualizeHomography:
    def __init__(self):
        self.hl = HomographyLogic(convert_to_pixels=True)
        print("Homography keys: ", self.hl.homographies.keys())
        self.h1: np.ndarray = self.hl.homographies["1"]
        self.h3: np.ndarray = self.hl.homographies["3"]

        self.j3_image = cv2.imread("data/images/jetson3.jpg")
        self.birds_eye_image = cv2.imread("data/images/fieldmodel.jpg")

    def visualize_homography(self, show_image: bool = False, save_image: bool = False) -> None :
        """
        This function will visualize the homography self.h1 onto self.birds_eye_view image
        """
        img_out = cv2.warpPerspective(self.j3_image, self.h3, (self.birds_eye_image.shape[1], self.birds_eye_image.shape[0]))

        if show_image:
            cv2.imshow("img_out", img_out)
            cv2.waitKey(0)
        if save_image:
            cv2.imwrite("data/images/jetson3_homography.jpg", img_out)

    def check_transformed_points(self) -> None:
        """
        Perform a homography on our original points and compare them to the transformed points.
        Ensure that the transformed points are the same as the original points, or very close to them.
        """
        # Get the original points
        c3, r3 = self.hl.get_real_world_coords(jetson_camera=self.hl.jetson3)

        # Get the transformed points
        transformed_points = np.array([])
        for point in c3:
            # Extend point from [x, y] to [x, y, 1]
            point = np.append(point, 1)
            point = self.hl.perform_homography(homography=self.h3, point=point)
            # convert from np.array to list and round to 1 decimal place
            point = [round(x, 1) for x in point]
            # Add to transformed_points (we will reshape later)
            transformed_points = np.append(transformed_points, point)

        transformed_points = transformed_points.reshape(-1, 3)  # Reshape to (n, 3)
        # Remove the last column (all 1's)
        transformed_points = transformed_points[:, :-1]
        # Convert numbers to not show in scientific notation

        # Copy numpy array to iterator to avoid changing original array
        transformed_points_iterator = iter(transformed_points.copy())


        # Compare the original points to the transformed points
        print("Original points: ", c3)
        print("real world points (targets): ", r3)
        print("Transformed points: ", transformed_points)

        # # Calculate the difference between the original and transformed points
        diff = r3 - transformed_points
        # Round all points to 1 decimal place
        diff = np.round(diff, 1)
        print("Difference between original and transformed points: ", diff)

        # Calculate the mean difference between the original and transformed points
        mean_diff = np.mean(diff, axis=0)
        print("Mean difference between original and transformed points: ", mean_diff)

        # Calculate the standard deviation of the difference between the original and transformed points
        std_diff = np.std(diff, axis=0)
        print("Standard deviation of the difference between original and transformed points: ", std_diff)

        # Creating another dictionary to print here so we can clearly identify points.
        iter_diff = iter(diff)
        labelling_points = self.get_labelled_dict(iter_diff, transformed_points_iterator, real_world_points=iter(r3), camera_points=iter(c3))

        # Print everything in the pd dataframe labelling_points
        pd.options.display.max_columns = None
        print(labelling_points)

    def get_labelled_dict(self, iter_diff: Iterator, transformed_points_iterator: Iterator, real_world_points: Iterator, camera_points: Iterator) -> pd.DataFrame:
        """
        This function will create a dictionary of the points and their labels.
        """
        # Create pandas dataframe
        keys = [key for key in self.hl.jetson3.__dict__ if self.hl.jetson3.__dict__[key] is not None]
        df = pd.DataFrame(columns=["key", "transformed_point", "difference", "real_world", "camera"])
        for count, (key, j, _diff, real, cam_p) in enumerate(zip(keys, transformed_points_iterator, iter_diff, real_world_points, camera_points)):
            if key is not None:
                df.loc[count] = [key, j.tolist(), _diff.tolist(), real, cam_p]

        return df


def main():
    visualize_homography = VisualizeHomography()
    # visualize_homography.visualize_homography(show_image=True, save_image=True)
    visualize_homography.check_transformed_points()


if __name__ == "__main__":
    main()
