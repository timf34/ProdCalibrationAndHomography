from typing import Tuple

HOMOGRAPHY_SAVE_DIR: str = "./data/homography_matrices/"


class CameraJetson1:
    def __init__(self):
        # Note: this definitely is not he cleanest way to do this, but it works for now
        #  What would be better is to have a utils file that stores all of this information as a dictionary maybe
        # Structure is (x_pixel_coords, y_pixel_coords)
        self.corner1: Tuple[int, int] = (807, 1005)
        self.corner2: Tuple[int, int] = (9, 646)
        self.bixBox1: Tuple[int, int] = (475, 886)
        self.bixBox2: Tuple[int, int] = (1062, 817)
        self.bixBox3: Tuple[int, int] = (406, 665)
        self.bigBox4: Tuple[int, int] = None
        self.box1: Tuple[int, int] = (295, 784)
        self.box2: Tuple[int, int] = (471, 772)
        self.box3: Tuple[int, int] = (262, 698)
        self.box4: Tuple[int, int] = (57, 673)
        self.goalPost1: Tuple[int, int] = (229, 748)  # They can be floats!
        self.goalPost2: Tuple[int, int] = (161, 716)
        self.semiCircle1: Tuple[int, int] = (775, 751)
        self.semiCircle2: Tuple[int, int] = (545, 698)
        self.boxParallel1: Tuple[int, int] = (1416, 899)
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (1470, 718)
        self.circle2: Tuple[int, int] = (1158, 670)
        self.halfway1: Tuple[int, int] = None

        # Real world camera coordinates
        self.real_world_x: float = -19.41
        self.real_world_y: float = -21.85
        self.real_world_z: float = 7.78


class CameraJetson3:
    def __init__(self):
        # Note: this definitely is not he cleanest way to do this, but it works for now
        #  What would be better is to have a utils file that stores all of this information as a dictionary maybe
        self.corner1: Tuple[int, int] = (1801, 298)
        self.corner2: Tuple[int, int] = (1805, 830)
        self.bixBox1: Tuple[int, int] = None
        self.bixBox2: Tuple[int, int] = (1408, 310)
        self.bixBox3: Tuple[int, int] = (983, 534)
        self.bigBox4: Tuple[int, int] = (1809, 595)
        self.box1: Tuple[int, int] = (1804, 362)
        self.box2: Tuple[int, int] = (1655, 355)
        self.box3: Tuple[int, int] = (1598, 456)
        self.box4: Tuple[int, int] = (1809, 471)
        self.goalPost1: Tuple[int, int] = (1806, 387)
        self.goalPost2: Tuple[int, int] = (1810, 427)
        self.semiCircle1: Tuple[int, int] = (1329, 347)
        self.semiCircle2: Tuple[int, int] = (1188, 419)
        self.boxParallel1: Tuple[int, int] = None
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (531, 323)
        self.circle2: Tuple[int, int] = (202, 398)
        self.halfway1: Tuple[int, int] = None

        # Real world camera coordinates
        self.real_world_x: float = 0.
        self.real_world_y: float = 86.16
        self.real_world_z: float = 7.85


class RealWorldPitchCoords:
    def __init__(self):
        self.corner1: Tuple[int, int] = (0, 64)
        self.corner2: Tuple[int, int] = (0, 0)
        self.bixBox1: Tuple[int, int] = (0, 54)
        self.bixBox2: Tuple[int, int] = (16, 54)
        self.bixBox3: Tuple[int, int] = (16, 10)
        self.bigBox4: Tuple[int, int] = (0, 10)
        self.box1: Tuple[int, int] = (0, 41)
        self.box2: Tuple[int, int] = (5, 41)
        self.box3: Tuple[int, int] = (5, 23)
        self.box4: Tuple[int, int] = (0, 23)
        self.goalPost1: Tuple[int, float] = (0, 34.5)  # They can be floats!
        self.goalPost2: Tuple[int, float] = (0, 19.5)
        self.semiCircle1: Tuple[int, int] = (16, 40)
        self.semiCircle2: Tuple[int, int] = (16, 24)
        self.boxParallel1: Tuple[int, int] = (16, 64)
        self.boxParallel2: Tuple[int, int] = None
        self.circle1: Tuple[int, int] = (51, 40)
        self.circle2: Tuple[int, int] = (51, 24)
        self.halfway1: Tuple[int, int] = None

    def convert_to_pixel_coords(self):
        """
        Converts the real world coordinates to pixel coordinates - multiplying by 1920, 1080
        """
        for key, value in self.__dict__.items():
            if value is not None:
                self.__dict__[key] = (value[0] * (1920/102), value[1] * (1080/64))


def main():
    # Test converting to pixels
    real_world_coords = RealWorldPitchCoords()
    print(real_world_coords.__dict__)
    real_world_coords.convert_to_pixel_coords()
    print(real_world_coords.__dict__)


if __name__ == "__main__":
    main()
