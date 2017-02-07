import math
import cv2
import numpy as np

WIDTH, HEIGHT = 400, 300

BLUE = (255, 0 , 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def get_frame():
    _, frame = capture.read()
    frame = frame[:, ::-1]  # flip video capture
    return cv2.resize(frame, (WIDTH, HEIGHT))


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


def get_convex_hull(contour):
    convex_hull = cv2.convexHull(contour)
    return convex_hull


def get_defect_points(contour):
    convexityDefects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    defects_count = 0
    defects = []
    if convexityDefects is not None:
        for i in range(convexityDefects.shape[0]):
            s, e, f, d = convexityDefects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                defects_count += 1
                defects.append(far)
    return defects_count, defects


def drawContour(img, contour, color):
    cv2.drawContours(img, [contour], 0, color, 2)


def drawPoint(img, middle, color):
    cv2.circle(img, middle, 3, color, -1)


def drawElementsToImage(img, contour, convex_hull, defect_points):
    drawContour(img, contour, BLUE)
    drawContour(img, convex_hull, GREEN)
    for point in defect_points:
        drawPoint(img, point, RED)


def main():
    while True:
        # get image from camera
        frame = get_frame()

        # perform operations on image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, thresh1 = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh1 = cv2.bitwise_not(thresh1)

        # finding contours
        hand_contour = get_hand_contour(thresh1)
        convex_hull = get_convex_hull(hand_contour)
        defects_count, defect_points = get_defect_points(hand_contour)
        print defects_count

        # draw transformation on image
        thresh_image = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        thresh_with_elements = np.copy(thresh_image)
        frame_with_elements = np.copy(frame)

        drawElementsToImage(thresh_with_elements, hand_contour, convex_hull, defect_points)
        drawElementsToImage(frame_with_elements, hand_contour, convex_hull, defect_points)

        top = np.hstack((frame, frame_with_elements))
        bottom = np.hstack((thresh_image, thresh_with_elements))
        image = np.vstack((top, bottom))

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
