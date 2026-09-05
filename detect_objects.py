import argparse

import cv2

from object_detector import create_debug_view, process_image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detect objects and display their centers and orientation axes."
    )
    parser.add_argument("-i", "--image", required=True, help="path to the input image")
    parser.add_argument(
        "--min-area",
        type=float,
        default=500,
        help="minimum contour area in pixels (default: 500)",
    )
    parser.add_argument("-o", "--output", help="optional path for the annotated image")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show all image-processing stages in one window",
    )
    parser.add_argument(
        "--debug-output",
        help="optional path for saving the visual debug panel",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.05,
        help="maximum shape difference for the same ID (default: 0.05)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
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

    if not detections:
        print("No objects found.")
    else:
        for detection in detections:
            center_x, center_y = detection["center"]
            print(
                "Object {}: id={}, center=({}, {}) px".format(
                    detection["object"], detection["id"], center_x, center_y
                )
            )

    if args.output and not cv2.imwrite(args.output, result):
        raise SystemExit("Unable to save result: {}".format(args.output))
    if args.output:
        print("Result saved to: {}".format(args.output))

    debug_view = None
    if args.debug or args.debug_output:
        debug_view = create_debug_view(
            image,
            raw_edges,
            closed_edges,
            result,
            len(detections),
            args.min_area,
            args.similarity_threshold,
        )

    if args.debug_output and not cv2.imwrite(args.debug_output, debug_view):
        raise SystemExit("Unable to save debug view: {}".format(args.debug_output))
    if args.debug_output:
        print("Debug view saved to: {}".format(args.debug_output))

    window_title = "Visual debug" if args.debug else "Detected objects"
    cv2.imshow(window_title, debug_view if args.debug else result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
