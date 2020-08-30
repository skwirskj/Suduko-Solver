import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt


def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plt_many_img(images, titles, rows=1, columns=2):
    for i, image in enumerate(images):
        plt.sublot(rows, columns, i + 1)
        plt.imshow(image, 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])


def show_digits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    show_img(np.concatenate(rows))


def convert_when_color(color, img):
    if len(color) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLORGRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLORGRAY2BGR)
    return img


def display_points(in_img, points, radius=5, color=(0, 0, 255)):
    img = in_img.copy()

    if len(color) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, color, -1)
    show_img(img)
    return img


def display_rects(in_img, rects, color=(0, 0, 255)):
    img = convert_when_color(color, in_img.copy())
    for rect in rects:
        img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), color)
    show_img(img)
    return img


def display_contours(in_img, contours, color=(0, 0, 255), thickness=2):
    img = convert_when_color(color, in_img.copy())
    img = cv2.drawContours(img, contours, -1, color, thickness)
    show_img(img)


def preprocess_board(img, dilate=True):
    # Applying gaussian blur
    processed_img = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    # Adaptive threshold with 11 nearest neighbor pixels
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Inverting the colors to have non-zero pixel values along the gridlines of the puzzle.
    processed_img = cv2.bitwise_not(processed_img, processed_img)

    if dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        processed_img = cv2.dilate(processed_img, kernel)

    return processed_img


def find_corners(img):
    contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    shape = contours[0]

    # Getting the index of the corner points
    btm_right_pt, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    top_left_pt, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    btm_left_pt, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in shape]), key=operator.itemgetter(1))
    top_right_pt, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in shape]), key=operator.itemgetter(1))

    return [shape[top_left_pt][0], shape[top_right_pt][0], shape[btm_right_pt][0], shape[btm_left_pt][0]]


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_wrap(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
    ])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, m, (int(side), int(side)))


def infer_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9

    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)
            p2 = ((i + 1) * side, (j + 1) * side)
            squares.append((p1, p2))
    return squares


def cut_from_rect(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


def scale_and_center(img, size, margin=0, background=0):
    h, w = img.shape[:2]

    def center_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = center_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = center_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
    img = inp_img.copy()
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:
                cv2.floodFill(img, mask, (x, y), 0)

                if img.item(y, x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def extract_digit(img, rect, size):
    digit = cut_from_rect(img, rect)

    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
    digit = cut_from_rect(digit, bbox)

    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return scale_and_center(digit, size)
    else:
        return np.zeros((size, size), np.uint8)


def get_digits(img, squares, size):
    digits = []
    img = preprocess_board(img.copy(), False)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def parse_grid(path):
    orig = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    processed = preprocess_board(orig)
    corners = find_corners(processed)
    cropped = crop_and_wrap(orig, corners)
    squares = infer_grid(cropped)
    digits = get_digits(cropped, squares, 28)
    show_digits(digits)


def main():
    parse_grid('sudukoBoard1.png')


if __name__ == '__main__':
    main()
