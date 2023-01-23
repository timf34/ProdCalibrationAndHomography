import numpy as np

from homgraphy_logic import HomographyLogic


def test_homography():
    h = HomographyLogic()

    # Ensure the shape of h.homographies is (3, 3)
    for key in h.homographies.keys():
        assert h.homographies[key].shape == (3, 3), "The homography matrix is not 3x3"

    # Ensure the shape of the transformed point is (3, 1)
    test_point1 = np.array([[1408], [310], [1.0]], dtype='float32')
    for key in h.homographies.keys():
        transformed_point = h.perform_homography(homography=h.homographies[key], point=test_point1)
        assert transformed_point.shape == (3, 1), "The transformed point is not 3x1"
