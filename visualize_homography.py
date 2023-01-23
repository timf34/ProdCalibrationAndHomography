import cv2
import numpy as np
from typing import Dict, List

from homgraphy_logic import compute_homographies


class VisualizeHomography:
    def __init__(self):
        self.h_dict: Dict = compute_homographies()
        print("h_dict.keys(): ", self.h_dict.keys())

        self.h1: List = self.h_dict["1"]
        self.h2: List = self.h_dict["3"]

        self.j3_image = cv2.imread("data/images/jetson3.jpg")
        self.birds_eye_image = cv2.imread("data/images/fieldmodel.jpg")

    def visualize_homography(self):
        """
        This function will visualize the homography self.h1 onto self.birds_eye_view image
        """
        img_out = cv2.warpPerspective(self.j3_image, self.h2, (self.birds_eye_image.shape[1], self.birds_eye_image.shape[0]))

        cv2.imshow("img_out", img_out)
        cv2.waitKey(0)

        #save image
        cv2.imwrite("data/images/jetson3_homography.jpg", img_out)



def main():
    visualize_homography = VisualizeHomography()
    visualize_homography.visualize_homography()


if __name__ == "__main__":
    main()
