import cv2 as cv
import numpy as np
import imutils


def edge_corner_detection(img):
    """
    Run an in-built edge detector and a corner detector
    :param img:
    :return: 2 output images (image with edges, image with corners)
    """
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    grey = np.float32(grey)
    block_size = 2
    aperture_size = 3
    k = 0.04
    corner_dst = cv.cornerHarris(grey, block_size, aperture_size, k)
    corner_dst = cv.dilate(corner_dst, None)
    # img[corner_dst > 0.01 * corner_dst.max()] = [0, 0, 255]
    edges_dst = cv.Canny(img, 100, 200)

    return corner_dst, edges_dst


def rotate(img):
    """
    Rotate the original image by 45-degrees
    """
    rotate_dst = imutils.rotate(img, angle=45)
    return rotate_dst


def scale(img):
    """
    Scale the original image by 1.5 in both the x and y-directions
    :param img:
    :return:
    """
    scale_dst = cv.resize(img, (0, 0), fx=1.5, fy=1.5)
    return scale_dst


def shear_x(img):
    """
    Shear the original image in the x-direction by 1.3
    :param img:
    :return:
    """
    # get the image shape
    rows, cols, dim = img.shape
    # transformation matrix for shearing applied to x-axis
    M = np.float32([[1, 1.3, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    # apply a perspective transformation to the image
    sheared_x_dst = cv.warpPerspective(img, M, (cols, rows))
    return sheared_x_dst


def shear_y(img):
    """
    Shear the original image in the y-direction by 1.3
    :param img:
    :return:
    """
    # get the image shape
    rows, cols, dim = img.shape
    # transformation matrix for shearing applied to y-axis
    m = np.float32([[1, 0, 0],
                    [1.3, 1, 0],
                    [0, 0, 1]])
    # apply a perspective transformation to the image
    sheared_y_dst = cv.warpPerspective(img, m, (cols, rows))
    return sheared_y_dst


def test():
    img = cv.imread("flower.jpg")
    flower_edge, flower_corner = edge_corner_detection(img)
    img = cv.imread("flower.jpg")
    flower_rotate_edge, flower_rotate_corner = edge_corner_detection(rotate(img))
    img = cv.imread("flower.jpg")
    flower_scale_edge, flower_scale_corner = edge_corner_detection(scale(img))
    img = cv.imread("flower.jpg")
    flower_shear_x_edge, flower_shear_x_corner = edge_corner_detection(shear_x(img))
    img = cv.imread("flower.jpg")
    flower_shear_y_edge, flower_shear_y_corner = edge_corner_detection(shear_y(img))

    results = [flower_edge, flower_corner, flower_rotate_edge, flower_rotate_corner, flower_scale_edge,
               flower_scale_corner, flower_shear_x_edge, flower_shear_x_corner, flower_shear_y_edge,
               flower_shear_y_corner]

    i = 0
    for image in results:
        cv.imwrite(str(i) + ".jpg", image)
        i += 1


test()
