import math
import cv2
import numpy as np
import requests
import time
from pyproj import Proj, transform

def calculateLeftCorner(latitude, longitude, side_length):
    # Преобразование широты и долготы в метры функция некорректная надо искать нормальную формулу
    # непонятно надо ли делить на 1000 - переводить метры в километры
    bias_lat = 1 / 111111
    bias_lon = 1 / (111111 * math.cos(math.radians(latitude)))
    side_length = (side_length / 1000) * 0.5
    left_corner_latitude = latitude - side_length * bias_lat
    left_corner_longitude = longitude - side_length * bias_lon
    print(left_corner_latitude, left_corner_longitude)
    return left_corner_latitude, left_corner_longitude

def calculateRightCorner(latitude, longitude, side_length):
    bias_lat = 1 / 111111
    bias_lon = 1 / (111111 * math.cos(math.radians(latitude)))
    side_length = (side_length / 1000) * 0.5
    right_corner_latitude = latitude - side_length * bias_lat
    right_corner_longitude = longitude - side_length * bias_lon
    print(right_corner_latitude, right_corner_longitude)
    return right_corner_latitude, right_corner_longitude

def calculatePixelPosition(latitude, longitude, level):
    map_size = 256 * 2 ** level
    latitude = min(max(latitude, -85.05112878), 85.05112878)
    longitude = min(max(longitude, 0.0), 180.0)
    sin_latitude = math.sin(latitude * math.pi / 180)
    pixel_x = ((longitude + 180) / 360) * map_size
    pixel_y = (0.5 - math.log((1 + sin_latitude) / (1 - sin_latitude)) / (4 * math.pi)) * map_size
    pixel_x = min(max(pixel_x, 0), map_size - 1)
    pixel_y = min(max(pixel_y, 0), map_size - 1)
    print(pixel_x, pixel_y)
    return (int(pixel_x), int(pixel_y))  #


def calculateTilePosition(pixel_position):
    tile_x = math.floor(pixel_position[0] / 256.0)
    tile_y = math.floor(pixel_position[1] / 256.0)
    return (int(tile_x), int(tile_y))


def calculateQuadKey(tile_position, level):
    tile_x = tile_position[0]
    tile_y = tile_position[1]
    quad_key = ""
    i = level
    while i > 0:
        digit = 0
        mask = 1 << (i - 1)
        if ((tile_x & mask) != 0):
            digit += 1
        if ((tile_y & mask) != 0):
            digit += 2
        quad_key += str(digit)
        i -= 1
    return quad_key


def downloadImage(quad_key):
    """ If you want to receive photo with labels - tiles/h, without - use tiles/a"""
    # url = 'http://h0.ortho.tiles.virtualearth.net/tiles/a' + quad_key + ".jpeg?g=131"
    url = 'http://h0.ortho.tiles.virtualearth.net/tiles/a' + quad_key + ".jpeg?g=131"
    print("Quad Key: " + quad_key + "\t| Tile URL: " + url)

    response = requests.get(url, stream=True)
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def downloadAerialImagery(upper_left_tile, lower_right_tile, filename):
    tiles = 0
    tilename = "tile"
    all_images = []
    for y in range(upper_left_tile[1], lower_right_tile[1] + 1):
        row_images = []
        for x in range(upper_left_tile[0], lower_right_tile[0] + 1):
            quad_key = calculateQuadKey((x, y), level)
            image = None
            while image is None or image.shape[0] == 0 or image.shape[1] == 0:
                image = downloadImage(quad_key)
                if image is None or image.shape[0] == 0 or image.shape[1] == 0:
                    print("Failed to download or load image for Quad Key:", quad_key)
                    # Add a delay before retrying to avoid continuous retries
                    time.sleep(1)
            tilename = tilename + str(tiles) + ".jpg"
            cv2.imwrite(tilename, image)
            tilename = "tile"
            row_images.append(image)
            tiles = tiles + 1
        row_images = np.concatenate(row_images, axis=1)
        all_images.append(row_images)
    all_images = np.concatenate(all_images, axis=0)
    cv2.imwrite(filename, all_images)
    print("-----------------------------------------------------------------------")
    print(" Total Number of Tiles Downloaded: ", tiles)
    print("-----------------------------------------------------------------------")
    return cv2.imread(filename)


def cropAerialImagery(image, p1, p2, filename):
    offset_x = p1[0] % 256
    offset_y = p1[1] % 256
    height = p2[1] - p1[1]
    width = p2[0] - p1[0]

    crop_img = image[offset_y:offset_y + height, offset_x:offset_x + width]

    # Check if crop_img is not empty
    if crop_img.shape[0] > 0 and crop_img.shape[1] > 0:
        cv2.imwrite(filename, crop_img)
        return cv2.imread(filename)
    else:
        print("Failed to crop image. Crop region may be invalid.")
        return None


def validateInput(p1, p2, t1, t2):
    upper_p = min(p1[1], p2[1])
    lower_p = max(p1[1], p2[1])
    left_p = min(p1[0], p2[0])
    right_p = max(p1[0], p2[0])
    p1 = (left_p, upper_p)
    p2 = (right_p, lower_p)

    upper_t = min(t1[1], t2[1])
    lower_t = max(t1[1], t2[1])
    left_t = min(t1[0], t2[0])
    right_t = max(t1[0], t2[0])
    t1 = (left_t, upper_t)
    t2 = (right_t, lower_t)
    return p1, p2, t1, t2


if __name__ == '__main__':
    print("-----------------------------------------------------------------------")
    print(" Enter Upper Left Corner Coordinates:")
    print(" ==================")
    print("\n Ground Resolution:")
    print(" ==================")
    # print(" Level ( 1 - 23 ): ", end="")
    # p1_latitude = 31.885375
    # p1_longitude = 34.960959
    # p2_latitude = 31.885385
    # p2_longitude = 34.960969
    center_latitude = 31.885375
    center_longitude = 34.960959
    p1_latitude, p1_longitude = calculateLeftCorner(center_latitude, center_longitude, 100)
    p2_latitude, p2_longitude = calculateRightCorner(center_latitude, center_longitude, 100)

    # level = int(input())
    "appropriate level - 16, 14 "
    level = int(16)
    print("\n Output Filename:")
    print(" ==================")
    print(" Filename (filename.jpg): ", end="")
    filename = "test" + str(input() + '.jpg')
    print("-----------------------------------------------------------------------")

    print(" Downloading and cropping satellite imagery...")
    print("-----------------------------------------------------------------------")
    p1 = calculatePixelPosition(p1_latitude, p1_longitude, level)
    p2 = calculatePixelPosition(p2_latitude, p2_longitude, level)
    print(p1,p2)
    t1 = calculateTilePosition(p1)
    t2 = calculateTilePosition(p2)
    p1, p2, t1, t2 = validateInput(p1, p2, t1, t2)

    image = downloadAerialImagery(t1, t2, filename)
    print(" Stitching tiles into one final image... ")
    print("-----------------------------------------------------------------------")
    image = cropAerialImagery(image, p1, p2, filename)
    print(" Finished processing satellite imagery.")
    print("-----------------------------------------------------------------------")
    print(" ", filename, " has ben created!")
    print("-----------------------------------------------------------------------\n\n")
