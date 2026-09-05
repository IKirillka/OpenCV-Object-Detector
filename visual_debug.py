import argparse

import cv2

from object_detector import create_debug_view, process_image


def main():
    parser = argparse.ArgumentParser(
        description="Show object detection stages, centers, and orientation axes."
    )
    parser.add_argument("-i", "--image", required=True, help="path to the input image")
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--similarity-threshold", type=float, default=0.05)
    parser.add_argument("-o", "--output", help="optional path for the debug panel")
    args = parser.parse_args()

    if args.min_area < 0:
        raise SystemExit("--min-area must be zero or greater")
    if args.similarity_threshold < 0:
        raise SystemExit("--similarity-threshold must be zero or greater")

    try:
        image, raw_edges, closed_edges, result, detections = process_image(
            args.image, args.min_area, args.similarity_threshold
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print("Objects processed: {}".format(len(detections)))
    for detection in detections:
        center_x, center_y = detection["center"]
        print(
            "Object {}: id={}, center=({}, {}) px".format(
                detection["object"], detection["id"], center_x, center_y
            )
        )

    debug_view = create_debug_view(
        image,
        raw_edges,
        closed_edges,
        result,
        len(detections),
        args.min_area,
        args.similarity_threshold,
    )
    if args.output and not cv2.imwrite(args.output, debug_view):
        raise SystemExit("Unable to save debug view: {}".format(args.output))
    if args.output:
        print("Debug view saved to: {}".format(args.output))

    cv2.imshow("Visual debug", debug_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
