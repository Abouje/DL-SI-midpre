import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, Tuple

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2: 检测+多目标跟踪+越线计数")
    parser.add_argument("--model", type=str, required=True, help="训练后的检测权重路径")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--output", type=str, default="outputs/tracking.mp4", help="输出视频路径")
    parser.add_argument("--log_csv", type=str, default="outputs/tracking_log.csv", help="逐帧轨迹日志")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--line", type=int, nargs=4, default=[100, 300, 1100, 300], help="越线计数线段 x1 y1 x2 y2")
    return parser.parse_args()


def side_of_line(pt: Tuple[float, float], a: Tuple[int, int], b: Tuple[int, int]) -> int:
    x, y = pt
    x1, y1 = a
    x2, y2 = b
    value = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def draw_line(img, a: Tuple[int, int], b: Tuple[int, int], count: int) -> None:
    cv2.line(img, a, b, (0, 255, 255), 2)
    cv2.putText(
        img,
        f"Crossed: {count}",
        (a[0], max(30, a[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_csv) or ".", exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    line_a = (args.line[0], args.line[1])
    line_b = (args.line[2], args.line[3])

    prev_side: Dict[int, int] = {}
    counted = set()
    crossed_count = 0
    frame_idx = 0
    history = defaultdict(list)

    with open(args.log_csv, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "track_id", "cls", "x1", "y1", "x2", "y2", "cx", "cy"])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                verbose=False,
            )[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                clss = result.boxes.cls.int().cpu().tolist()

                for box, tid, cls_id in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box.tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    side = side_of_line((cx, cy), line_a, line_b)

                    if tid in prev_side and tid not in counted:
                        if side != 0 and prev_side[tid] != 0 and side != prev_side[tid]:
                            crossed_count += 1
                            counted.add(tid)

                    prev_side[tid] = side
                    history[tid].append((cx, cy))

                    csv_writer.writerow([frame_idx, tid, cls_id, x1, y1, x2, y2, cx, cy])

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID {tid} C{cls_id}",
                        (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 200, 0),
                        2,
                    )

            draw_line(frame, line_a, line_b, crossed_count)
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"处理完成: {args.output}")
    print(f"越线总数: {crossed_count}")
    print(f"轨迹日志: {args.log_csv}")


if __name__ == "__main__":
    main()
