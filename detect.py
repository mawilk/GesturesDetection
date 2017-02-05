import math
import cv2
import numpy as np

WIDTH, HEIGHT = 830, 630


def get_frame():
    _, frame = capture.read()
    frame = frame[:, ::-1]  # flip video capture
    return cv2.resize(frame, (400, 300))


def get_hand_contour(image):
    image_copy = np.copy(image)
    contours, hierarchy = cv2.findContours(image_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area_index, max_area = 0, 0
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_area_index = index
    contour = contours[max_area_index]
    return contour


def get_image_with_convex_hull(image, contour):
    image_copy = np.copy(image)
    convex_hull = cv2.convexHull(contour)
    cv2.drawContours(image_copy, [contour], 0, (255, 255, 255), 2)
    cv2.drawContours(image_copy, [convex_hull], 0, (255, 255, 255), 2)
    return image_copy


def get_image_with_defect_points(image, contour):
    image_copy = np.zeros(image.shape)
    defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    defects_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            defects_count += 1
            cv2.circle(image_copy, far, 3, [255, 255, 255], -1)
        cv2.line(image_copy, start, end, [255, 255, 255], 2)
    return defects_count, image_copy


def main():
    while True:
        # create image to show
        image = np.zeros((HEIGHT, WIDTH), np.uint8)

        # get image from camera
        frame = get_frame()

        # perform operations on image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # finding contours
        hand_contour = get_hand_contour(thresh1)
        image_with_convex_hull = get_image_with_convex_hull(thresh1, hand_contour)
        defects_count, image_with_defect_points = get_image_with_defect_points(image_with_convex_hull, hand_contour)

        # draw transformation on image
        image[10:10 + 300, 10:10 + 400] = gray
        image[320:320 + 300, 10:10 + 400] = image_with_convex_hull
        image[10:10 + 300, 420:420 + 400] = thresh1
        image[320:320 + 300, 420:420 + 400] = image_with_defect_points

        print defects_count

        # show image
        cv2.imshow('frame', image)

        # quit application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    main()

    # cleanup
    capture.release()
    cv2.destroyAllWindows()
