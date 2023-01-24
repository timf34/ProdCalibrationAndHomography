import cv2
import pandas as pd
import numpy as np
from typing import Iterator

from homgraphy_logic import HomographyLogic

np.set_printoptions(suppress=True)  # Suppress scientific notation
pd.options.display.max_columns = None  # Display all pandas columns


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

    def get_transformed_points(self, camera_points: np.ndarray) -> np.ndarray:
        """
        Transforms the camera points to the real world points.
        :param camera_points: The camera points to transform - shape is (n, 2)
        :return: The transformed points - shape is (n, 2)
        """
        camera_points_3d = np.c_[camera_points, np.ones(len(camera_points))]
        transformed_points = np.array([])

        for point in camera_points_3d:
            point = self.hl.perform_homography(homography=self.h3, point=point)
            point = [round(x, 1) for x in point]  # Convert to list and round to 1 decimal place
            transformed_points = np.append(transformed_points, point)

        return transformed_points.reshape(-1, 3)[:, :-1]  # Reshape to (n, 3) and remove the last column (all 1's)

    def print_stats(self, homography_dataframe: pd.DataFrame) -> None:
        """
        This function will print the stats of the homography dataframe -> Difference, mean, std dev
        :param homography_dataframe: Columns are "key", "transformed_point", "difference", "real_world", "camera_coords"
        """
        # print("Difference between original and transformed points: ", homography_dataframe["difference"].values)
        # print("Mean difference between original and transformed points: ", np.mean(homography_dataframe["difference"].values, axis=0))
        # print("Standard deviation of the difference between original and transformed points: ", np.std(homography_dataframe["difference"].values, axis=0))

        # We first need to convert the diff column to a np array
        diff = np.array([np.array(x) for x in homography_dataframe["difference"].values])
        print("Mean difference between original and transformed points: ", np.mean(diff, axis=0))
        print("Standard deviation of the difference between original and transformed points: ", np.std(diff, axis=0))

    def get_labelled_dict(self, iter_diff: Iterator, transformed_points_iterator: Iterator, real_world_points: Iterator,
                          camera_points: Iterator) -> pd.DataFrame:
        """
        This function will create a dictionary of the points and their labels.
        """
        # Create pandas dataframe
        keys = [key for key in self.hl.jetson3.__dict__ if self.hl.jetson3.__dict__[key] is not None]
        df = pd.DataFrame(columns=["key", "transformed_point", "difference", "real_world", "camera_coords"])
        for count, (key, j, _diff, real, cam_p) in enumerate(zip(keys, transformed_points_iterator, iter_diff,
                                                                 real_world_points, camera_points)):
            if key is not None:
                df.loc[count] = [key, j.tolist(), _diff.tolist(), real, cam_p]

        return df

    def check_transformed_points(self) -> None:
        """
        Perform a homography on our original points and compare them to the transformed points.
        Ensure that the transformed points are the same as the original points, or very close to them.
        """
        # Get the original points
        c3, r3 = self.hl.get_real_world_coords(jetson_camera=self.hl.jetson3)
        # Get the transformed points
        transformed_points = self.get_transformed_points(camera_points=c3)

        diff = r3 - transformed_points  # Calculate the difference between the original and transformed points
        diff = np.round(diff, 1)  # Round to 1 decimal place

        labelling_points = self.get_labelled_dict(iter(diff), iter(transformed_points.copy()),
                                                  real_world_points=iter(r3), camera_points=iter(c3))

        self.print_stats(labelling_points)

        print(labelling_points)


def main():
    visualize_homography = VisualizeHomography()
    # visualize_homography.visualize_homography(show_image=True, save_image=True)
    visualize_homography.check_transformed_points()


if __name__ == "__main__":
    main()
