from PIL import Image, ImageDraw
import numpy as np
import math
import cv2


def flow_chart(width, optical_flow=True):
    W_2 = width / 2

    x = np.arange(-W_2, W_2, 1)
    y = np.arange(-W_2, W_2, 1)
    yy, xx = np.meshgrid(y, x)
    r = np.sqrt(xx**2 + yy**2)
    normed_r = r / W_2

    angle = np.arctan2(xx, yy)
    angle = 360 - (angle * 180 / np.pi)

    if optical_flow:
        # Radius as value, which maps optical flow's magnitude
        hsv_color_wheel = np.dstack([angle, np.ones((width, width)), normed_r * 255]) 
    else:
        # Radius as saturation, which is a more conventional HSV wheel
        hsv_color_wheel = np.dstack([angle, normed_r, np.full((width, width), 255)])

    hsv_color_wheel[r >= W_2] = 0


    hsv_color_wheel = np.asarray(hsv_color_wheel, dtype=np.float32)
    hsv_color_wheel = cv2.cvtColor(hsv_color_wheel, cv2.COLOR_HSV2RGB)

    hsv_color_wheel_flat = hsv_color_wheel.reshape(width * width, 3)
    hsv_color_wheel_flat_tuple = [tuple(v) for v in hsv_color_wheel_flat]
    hsv_color_wheel_img = Image.new("RGB", (width, width))
    hsv_color_wheel_img.putdata(hsv_color_wheel_flat_tuple)

    return np.array(hsv_color_wheel_img)
    

wh = 200
hsv_color_wheel_img = flow_chart(wh, optical_flow=True)

# cv2.imshow("", hsv_color_wheel_img)
# cv2.waitKey()
# raise

# Generate some shifted boxes to demonstrate optical flow
box_sz = wh / 5
box_sz_2 = 2 * wh / 5
box_sz_3 = 3 * wh / 5
dim = (wh, wh)
img1 = Image.new("L", dim)
img2 = Image.new("L", dim)

shift = 5
draw = ImageDraw.Draw(img1)
draw.ellipse((wh/5+10, wh/5+10, 4*wh/5-10, 4*wh/5-10), 255)
draw.ellipse((wh/5+20, wh/5+20, 4*wh/5-20, 4*wh/5-20), 0)
draw = ImageDraw.Draw(img2)
draw.ellipse((wh/5+10 - shift, wh/5+10 - shift, 4*wh/5-10 + shift, 4*wh/5-10 + shift), 255)
draw.ellipse((wh/5+20 - shift, wh/5+20 - shift, 4*wh/5-20 + shift, 4*wh/5-20 + shift), 0)

img1_arr = np.array(img1)
img2_arr = np.array(img2)

# Run optical flow and visualize two possible ways
for method in [1, 2]:
    flow = cv2.calcOpticalFlowFarneback(img1_arr, img2_arr, None, 0.5, 3, 40, 3, 5, 1.2, 0)

    # Convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create an HSV image
    hsv = np.zeros((img1.size[0], img1.size[1], 3))
    hsv[:,:,1] = 1  # Full saturation
    # Set hue from the flow direction
    if method == 1:
        hsv[:,:,0] = ang * (180 / np.pi / 2)
    else:
        hsv[:,:,0] = 360 - (ang * (180 / np.pi))
    # Set value from the flow magnitude
    hsv[:,:,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # Convert HSV to int32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    print(rgb_flow.shape)
    raise

    # Convert to an image
    rgb_flow_flat = rgb_flow.reshape(rgb_flow.shape[0] * rgb_flow.shape[1], 3)
    rgb_flow_flat_tuple = [tuple(v) for v in rgb_flow_flat]
    flow_img = Image.new("RGB", img1.size)
    flow_img.putdata(rgb_flow_flat_tuple)

    analysis_img = Image.new("RGB", (img1.size[0] * 4, img2.size[1]))
    analysis_img.paste(img1, (0, 0))
    analysis_img.paste(img2, (img1.size[0], 0))
    analysis_img.paste(flow_img, (img1.size[0] * 2, 0))
    print(img1.size[0])
    raise

    temp = np.array(flow_img)
    print(temp.dtype, temp.shape)
    print(hsv_color_wheel_img.dtype, hsv_color_wheel_img.shape)
    analysis_img.paste(hsv_color_wheel_img, (img1.size[0] * 3, 0))


    analysis_img = np.array(analysis_img)
    print("Original visualization method:" if method == 1 else "Proposed visualization method:")
    cv2.imshow("image", analysis_img)
    cv2.waitKey(10000)