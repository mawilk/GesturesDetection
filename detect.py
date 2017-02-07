import math
import cv2
import numpy as np

WIDTH, HEIGHT = 400, 300

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


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
    convexity_defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
    defects_count = 0
    defects = []
    tips = []
    if convexity_defects is not None:
        for i in range(convexity_defects.shape[0]):
            s, e, f, d = convexity_defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
            if angle <= 130:
                defects_count += 1
                defects.append(far)
                tips.append(end)
                tips.append(start)

    return defects_count, defects, tips


def limit_finger_tips(potential_tips, center):
    result = []
    if len(potential_tips) <= 0:
        return result

    potential_tips = [item for item in potential_tips if is_above(item, center, 100)]

    furthest = max(potential_tips, key=lambda pt:get_distance(pt, center))
    hand_radius = get_distance(furthest, center) / 2

    potential_tips = [item for item in potential_tips if get_distance(item, center) > hand_radius]

    potential_tips = sorted(potential_tips, key=lambda pt: pt[0])

    result.append(potential_tips[0])

    for i in xrange(len(potential_tips) - 1):
        dist = get_distance(potential_tips[i], potential_tips[i + 1])
        if dist > 15:
            result.append(potential_tips[i + 1])
    return result


def get_middle(p1, p2):
    r = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    return r


def is_above(p1, p2, margin=0):
    return p1[1] < p2[1] + margin


def get_hand_middle(contour):
    m = cv2.moments(contour)
    x = int(m['m10'] / m['m00'])
    y = int(m['m01'] / m['m00'])
    return x, y


def get_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def draw_contour(img, contour, color):
    cv2.drawContours(img, [contour], 0, color, 2)


def draw_point(img, middle, color):
    cv2.circle(img, middle, 4, color, -1)


def draw_elements_to_image(img, contour, convex_hull, defect_points, middle, tips):
    draw_contour(img, contour, BLUE)
    draw_contour(img, convex_hull, GREEN)
    for point in defect_points:
        draw_point(img, point, RED)
    draw_point(img, middle, YELLOW)
    for i in xrange(len(tips)):
        cv2.putText(img, str(i + 1), tips[i], cv2.FONT_HERSHEY_COMPLEX, 1, 80)
        draw_point(img, tips[i], YELLOW)


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
        defects_count, defect_points, tips = get_defect_points(hand_contour)
        hand_middle = get_hand_middle(hand_contour)
        tips = limit_finger_tips(tips, hand_middle)

        # draw transformation on image
        thresh_image = cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR)
        thresh_with_elements = np.copy(thresh_image)
        frame_with_elements = np.copy(frame)

        draw_elements_to_image(thresh_with_elements, hand_contour, convex_hull, defect_points, hand_middle, tips)
        draw_elements_to_image(frame_with_elements, hand_contour, convex_hull, defect_points, hand_middle, tips)

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
