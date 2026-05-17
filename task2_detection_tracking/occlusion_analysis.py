import argparse
import os

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2: 遮挡片段可视化导出")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/occlusion_frames")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {args.video}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    for i in range(args.num_frames):
        ok, frame = cap.read()
        if not ok:
            break

        frame_id = args.start_frame + i
        result = model.track(frame, persist=True, tracker=args.tracker, verbose=False)[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.int().cpu().tolist()
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 220, 50), 2)
                cv2.putText(
                    frame,
                    f"ID {tid}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (50, 220, 50),
                    2,
                )

        save_path = os.path.join(args.output_dir, f"frame_{frame_id:06d}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"已导出: {save_path}")

    cap.release()


if __name__ == "__main__":
    main()
