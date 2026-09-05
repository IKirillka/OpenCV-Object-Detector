import cv2
import imutils
import numpy as np
from imutils import contours, perspective


TARGET_SIZE = (640, 480)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to open image: {}".format(image_path))
    return cv2.resize(image, TARGET_SIZE)


def create_edge_maps(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    raw_edges = cv2.Canny(gray, 50, 100)
    closed_edges = cv2.dilate(raw_edges, None, iterations=1)
    closed_edges = cv2.erode(closed_edges, None, iterations=1)
    return raw_edges, closed_edges


def find_objects(edge_map):
    found = cv2.findContours(
        edge_map.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return imutils.grab_contours(found)


def midpoint(point_a, point_b):
    return (
        (point_a[0] + point_b[0]) * 0.5,
        (point_a[1] + point_b[1]) * 0.5,
    )


def remove_nested_contours(found_contours):
    outer_contours = []
    for contour in sorted(found_contours, key=cv2.contourArea, reverse=True):
        center = cv2.minAreaRect(contour)[0]
        is_nested = any(
            cv2.pointPolygonTest(
                cv2.boxPoints(cv2.minAreaRect(outer_contour)), center, False
            )
            >= 0
            for outer_contour in outer_contours
        )
        if not is_nested:
            outer_contours.append(contour)
    return outer_contours


def assign_class_id(contour, object_classes, similarity_threshold):
    for class_index, class_contours in enumerate(object_classes):
        best_score = min(
            cv2.matchShapes(contour, sample, cv2.CONTOURS_MATCH_I1, 0.0)
            for sample in class_contours
        )
        if best_score <= similarity_threshold:
            class_contours.append(contour)
            return class_index + 1, best_score

    object_classes.append([contour])
    return len(object_classes), 0.0


def annotate_objects(
    image, found_contours, min_area=500, similarity_threshold=0.05
):
    result = image.copy()
    valid_contours = [
        contour
        for contour in found_contours
        if cv2.contourArea(contour) >= min_area
    ]
    valid_contours = remove_nested_contours(valid_contours)

    if valid_contours:
        valid_contours, _ = contours.sort_contours(valid_contours)

    detections = []
    object_classes = []
    for object_index, contour in enumerate(valid_contours, start=1):
        class_id, similarity_score = assign_class_id(
            contour, object_classes, similarity_threshold
        )
        rectangle = cv2.minAreaRect(contour)
        box = perspective.order_points(cv2.boxPoints(rectangle))

        top_left, top_right, bottom_right, bottom_left = box
        top_mid = midpoint(top_left, top_right)
        bottom_mid = midpoint(bottom_left, bottom_right)
        left_mid = midpoint(top_left, bottom_left)
        right_mid = midpoint(top_right, bottom_right)

        center_x = int(round(rectangle[0][0]))
        center_y = int(round(rectangle[0][1]))
        detections.append(
            {
                "object": object_index,
                "id": class_id,
                "center": (center_x, center_y),
                "box": box.astype("int").tolist(),
                "similarity": similarity_score,
            }
        )

        cv2.drawContours(result, [box.astype("int")], -1, (0, 255, 0), 2)
        cv2.line(result, tuple(map(int, top_mid)), tuple(map(int, bottom_mid)), (255, 0, 255), 2)
        cv2.line(result, tuple(map(int, left_mid)), tuple(map(int, right_mid)), (255, 0, 255), 2)
        cv2.circle(result, (center_x, center_y), 6, (0, 255, 255), -1)
        cv2.putText(
            result,
            "ID-{} ({}, {})".format(class_id, center_x, center_y),
            (center_x + 10, max(center_y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

    return result, detections


def process_image(image_path, min_area=500, similarity_threshold=0.05):
    image = load_image(image_path)
    raw_edges, closed_edges = create_edge_maps(image)
    found_contours = find_objects(closed_edges)
    result, detections = annotate_objects(
        image, found_contours, min_area, similarity_threshold
    )
    return image, raw_edges, closed_edges, result, detections


def _debug_tile(title, image, size=(320, 240)):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(image, size)
    cv2.rectangle(tile, (0, 0), (size[0], 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def create_debug_view(
    image,
    raw_edges,
    closed_edges,
    result,
    object_count,
    min_area,
    similarity_threshold,
):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    tiles = [
        _debug_tile("1. Original 640x480", image),
        _debug_tile("2. Grayscale", gray),
        _debug_tile("3. Gaussian blur 7x7", blurred),
        _debug_tile("4. Canny 50/100", raw_edges),
        _debug_tile("5. Dilate + erode", closed_edges),
        _debug_tile(
            "6. Objects: {} | area >= {} | similarity <= {:.3f}".format(
                object_count, int(min_area), similarity_threshold
            ),
            result,
        ),
    ]
    return np.vstack((np.hstack(tiles[:3]), np.hstack(tiles[3:])))
