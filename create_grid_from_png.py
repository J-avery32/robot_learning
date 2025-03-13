import PIL
import PIL.Image
import numpy as np
np.set_printoptions(linewidth=130)


def grid_from_image(filename: str):
    im =PIL.Image.open(filename)

    lake_map = np.zeros((25,25))

    for x in range(25):
        for y in range(25):
            pixel = im.getpixel((x*24+12, y*24+12))
            if pixel[0] >= 200:
                # land
                lake_map[x][y] = 1
            else:
                # water
                lake_map[x][y] = 0
    return np.flip(lake_map, axis=1)

# print(grid_from_image("lake_obstacle.png"))
