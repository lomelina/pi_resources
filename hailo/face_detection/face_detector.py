import numpy as np


class FaceDetector:
    SCALES_INFO = [
        {"stride": 8, "cls_name": "scrfd_2_5g/conv42", "reg_name": "scrfd_2_5g/conv43", "base_size": 64, "keypoints": "scrfd_2_5g/conv44"},
        {"stride": 16, "cls_name": "scrfd_2_5g/conv49", "reg_name": "scrfd_2_5g/conv50", "base_size": 128, "keypoints": "scrfd_2_5g/conv51"},
        {"stride": 32, "cls_name": "scrfd_2_5g/conv55", "reg_name": "scrfd_2_5g/conv56", "base_size": 256, "keypoints": "scrfd_2_5g/conv57"},
    ]
    INPUT_SIZE = 640
    SCORE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    NORMALIZED_EYE_SIZE = (256, 64)

    def __init__(
        self,
        scales_info=None,
        input_size=None,
        score_threshold=None,
        nms_threshold=None,
        normalized_eye_size=None,
    ):
        self.scales_info = scales_info if scales_info is not None else self.SCALES_INFO
        self.input_size = input_size if input_size is not None else self.INPUT_SIZE
        self.score_threshold = score_threshold if score_threshold is not None else self.SCORE_THRESHOLD
        self.nms_threshold = nms_threshold if nms_threshold is not None else self.NMS_THRESHOLD
        self.normalized_eye_size = (
            normalized_eye_size if normalized_eye_size is not None else self.NORMALIZED_EYE_SIZE
        )
        self.anchors_cache = self.build_anchors_cache()

    @staticmethod
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        if area1 + area2 - inter == 0:
            return 0.0
        return inter / (area1 + area2 - inter)

    @staticmethod
    def build_anchors(H, W, stride, base_size):
        ratios = [1.0]
        scales = [1.0, 2.0**0.5]
        anchors = []
        for y in range(H):
            for x in range(W):
                for ratio in ratios:
                    for sc in scales:
                        anchor_w = base_size * sc / ratio**0.5
                        anchor_h = base_size * sc * ratio**0.5
                        anchor_x = x * stride
                        anchor_y = y * stride
                        anchors.append((anchor_x, anchor_y, anchor_w, anchor_h))
        return anchors, len(ratios) * len(scales)

    def build_anchors_cache(self):
        anchors_cache = {}
        for scale in self.scales_info:
            H = self.input_size // scale["stride"]
            W = self.input_size // scale["stride"]
            cache_key = (scale["stride"], scale["base_size"], H, W)
            anchors_cache[cache_key] = self.build_anchors(
                H, W, scale["stride"], scale["base_size"]
            )
        return anchors_cache

    def apply_nms(self, detections):
        if not detections:
            return []
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []
        while detections:
            keep.append(detections[0])
            detections = [
                d for d in detections[1:]
                if self.compute_iou(keep[-1][:4], d[:4]) < self.nms_threshold
            ]
        return keep

    def decode_scale_outputs(self, outputs, scale, anchors, anchors_per_cell):
        detections = []
        stride = scale["stride"]
        cls = outputs[scale["cls_name"]]
        reg = outputs[scale["reg_name"]]
        kps = outputs[scale["keypoints"]]

        H, W, num_anchors = cls.shape
        if num_anchors != anchors_per_cell:
            return detections
        reg = reg.reshape(H, W, num_anchors, 4)
        kps = kps.reshape(H, W, num_anchors, 10)

        for y in range(H):
            for x in range(W):
                for a in range(num_anchors):
                    score = cls[y, x, a]
                    if score < self.score_threshold:
                        continue
                    dx, dy, dw, dh = reg[y, x, a]
                    kp = kps[y, x, a]
                    anchor_idx = y * W * anchors_per_cell + x * anchors_per_cell + a
                    ax, ay, aw, ah = anchors[anchor_idx]
                    x1 = ax - dx * stride
                    y1 = ay - dy * stride
                    x2 = ax + dw * stride
                    y2 = ay + dh * stride
                    x1 = max(0, min(self.input_size, x1))
                    y1 = max(0, min(self.input_size, y1))
                    x2 = max(0, min(self.input_size, x2))
                    y2 = max(0, min(self.input_size, y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    keypoints = []
                    for i in range(0, 10, 2):
                        kpx = ax + kp[i] * stride
                        kpy = ay + kp[i + 1] * stride
                        kpx = max(0, min(self.input_size, kpx))
                        kpy = max(0, min(self.input_size, kpy))
                        keypoints.append((kpx, kpy))
                    detections.append([x1, y1, x2, y2, score, keypoints])
        return detections

    def find_faces(self, outputs):
        detections = []
        for scale in self.scales_info:
            cls = outputs[scale["cls_name"]]
            H, W, _ = cls.shape
            cache_key = (scale["stride"], scale["base_size"], H, W)
            anchors, anchors_per_cell = self.anchors_cache[cache_key]
            detections.extend(
                self.decode_scale_outputs(outputs, scale, anchors, anchors_per_cell)
            )
        detections = self.apply_nms(detections)
        if len(detections) > 1:
            print("Warning: more than one face detected, using the largest one.")
        return detections
