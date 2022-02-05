# Author: Michael Xu
# LeBeau Group, MIT


import numpy as np
import cv2
from scipy.ndimage import convolve
from skimage.morphology import disk


def transform_shift(shift, step):
    shift_scaled = shift*step
    magnitude = np.max(np.abs(shift_scaled))
    noise = np.random.normal(0, magnitude / 1000, shift_scaled.shape)
    # noise stddev. is 10% of the current
    # max shift magnitude
    noised_shift = shift_scaled + noise
    rotated_shift = rotate2D(noised_shift, np.pi/6)
    x = rotated_shift[0]
    y = rotated_shift[1]
    return x,y


def rotate2D(coord, radians):
    rot_mat = np.array([[np.cos(radians), -np.sin(radians)],
                        [np.sin(radians), np.cos(radians)]])
    rotated = np.dot(rot_mat, coord)
    return rotated


class FlucamSimulator_sim:
    ## beam position should never be accessed outside of the instance
    ## flu image is dynamically dependent on beam position in the instance
    ## also should never be called externally

    def __init__(self, img_size, source_size, mode, step):
        self._step_size = step*1040/255
        self._size = img_size
        self._displaysize = 300
        self._sourcesize = source_size
        self._beam_pos = np.random.randint(0, 1040, 2).astype(np.float64)
        self._start_pos = np.copy(self._beam_pos)
        self._image = np.zeros((self._displaysize, self._displaysize), dtype=np.uint8)
        self._mode = mode
        self._offset = 0
        self._radius = 9

        if self._mode == "pixel":
            self.template = np.ones((1, 1), dtype=np.uint8)
            self._offset = 0
        elif self._mode == "disk":
            self.template = disk(self._radius, dtype=int)
            self._offset = 10
        self.update_fluscreen()

    @property
    def get_size(self):
        return self._size

    @property
    def get_pos(self):
        return self._beam_pos

    def update_drift_noise(self):
        if all(self.drift_vel == np.array([0,0])):
            rounded_shift = np.array([0,0])
        else:
            magnitude = np.max(np.abs(self.drift_vel))
            noise = np.random.normal(0, magnitude / 20, self.drift_vel.shape)
            # noise stddev. is 10% of the current
            # max shift magnitude
            noised_shift = self.drift_vel + noise
            rounded_shift = np.around(noised_shift, decimals=0).astype(dtype=int)
        return rounded_shift

    def shift_beam(self, shifts):
        shift_x, shift_y = transform_shift(shifts, self._step_size)
        self._beam_pos += np.array([shift_x, shift_y])
        self.drift_vel = self.update_drift_noise()

    def read_fluscreen(self):
        beam_status = self.check_beam()
        if beam_status[0]:
            self.update_fluscreen()
            # COM is not measured in the simulator, but rather the explicit coordinates are used
            # COM + sharp disk offers no difference than raw coordinates
            return {"beam": True, "pos": self.get_pos}
        else:
            return {"beam": False, "pos": self.get_pos}

    def check_beam(self):
        beam_pos = self._beam_pos
        if any(beam_pos>= np.array([self._size-self._offset, self._size-self._offset])):
            return False, beam_pos
        elif any(beam_pos< np.array([0+self._offset, 0+self._offset])):
            return False, beam_pos
        else:
            return True, beam_pos

    def update_fluscreen(self):
        self.reset_fluscreen()
        beam = np.round(self.downsize_coords_display(self._beam_pos), decimals=0).astype(np.int)
        self._image[beam[0], beam[1]] = 255
        self._image = convolve(self._image, self.template)
        print("beam: ", beam)

        return self._image

    def reset_fluscreen(self):
        self._image = np.zeros((self._displaysize+1, self._displaysize+1), dtype=np.uint8)

    def render_fluscreen(self, goal, mode):
        screen = np.stack([self._image, self._image, self._image], axis=2)
        goal_screen = self.upsize_coords_display((goal))
        start_screen = self.downsize_coords_display((self._start_pos))
        screen[goal_screen[0], goal_screen[1]] = np.array([0, 255, 0])
        screen[start_screen[0], start_screen[1]] = np.array([0, 0, 255])
        pos_edge = self.upsize_coords_display(np.array([127,127]))
        screen[pos_edge[0], pos_edge[1]] = np.array([255, 255, 0])
        neg_edge = self.upsize_coords_display(np.array([-127,-127]))
        screen[neg_edge[0], neg_edge[1]] = np.array([0, 255, 255])
        scale_percent = 300  # percent of original size
        width = int(screen.shape[1] * scale_percent / 100)
        height = int(screen.shape[0] * scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(screen, dim, interpolation=cv2.INTER_AREA)
        if mode == 'human':
            cv2.imshow("test_render", image)
            cv2.waitKey(1)

    def upsize_coords_display(self, down_coords):
        scale_new = down_coords*(self._displaysize/self._sourcesize)
        pos_new = scale_new + np.array([self._displaysize/2, self._displaysize/2])
        coords = np.round(pos_new, decimals=0)

        return coords.astype(np.int)

    def downsize_coords_display(self, down_coords):
        coords = np.round((down_coords )/(self._size/self._displaysize)).astype(np.int)

        return coords