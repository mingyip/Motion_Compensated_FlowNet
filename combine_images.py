import numpy as np
import cv2 as cv 

for i in range(100, 201):

    # print(i)
    image_name="results/test01/img_{:07d}.png".format(i)
    image_path_name="results/test01/path_{:07d}.png".format(i)
    image_gt_path_name="results/test01/gt_path_{:07d}.png".format(i)
    image_final_name="results/test01/final_{:07d}.png".format(i)

    img = cv.imread(image_name)
    tr_img = cv.imread(image_path_name)
    gt_img = cv.imread(image_gt_path_name)
    # tr_img = cv.resize(tr_img, (240, 320), interpolation = cv.INTER_AREA)

    canvas = np.zeros((992, 1680, 3))
    canvas[0:512, 0:1680] = img
    canvas[512:992, 0:640] = tr_img
    canvas[512:992, 640:1280] = gt_img

    cv.putText(canvas, "Frame: " + str(i), (340, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv.imwrite(image_final_name, canvas)
