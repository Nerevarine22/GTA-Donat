from dataclasses import dataclass
from time import time

import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO

import config


Frame = NDArray[np.uint8]
Color = tuple[int, int, int]
Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Assignment:
    nickname: str
    color: Color
    assigned_at: float


@dataclass(frozen=True, slots=True)
class TrackedPerson:
    track_id: int
    head: Point
    assignment: Assignment
    box: Box


@dataclass(slots=True)
class HeadPosition:
    point: Point
    frame_number: int


class PersonTracker:
    def __init__(self) -> None:
        self._model = YOLO(config.MODEL_WEIGHTS)
        self.assignments: dict[int, Assignment] = {}
        self._heads: dict[int, HeadPosition] = {}
        self._frame_number = 0
        self._donor_cursor = 0

    def update(self, frame: Frame) -> list[TrackedPerson]:
        self._frame_number += 1
        result = self._model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[0],
            conf=config.DETECTION_CONFIDENCE,
            imgsz=config.INFERENCE_SIZE,
            device=config.CUDA_DEVICE,
            quantize=16,
            verbose=False,
        )[0]

        self._heads = {
            track_id: head
            for track_id, head in self._heads.items()
            if self._frame_number - head.frame_number <= config.EMA_MAX_GAP_FRAMES
        }

        boxes = result.boxes
        people: list[TrackedPerson] = []
        if boxes is None or not boxes.is_track:
            return people

        # Copy the tracked boxes and their IDs together to avoid two GPU transfers.
        boxes = boxes.cpu()
        track_ids = boxes.id.numpy().astype(int).tolist()
        visible_labels = {
            self.assignments[track_id].nickname
            for track_id in track_ids
            if track_id in self.assignments
        }
        for box, track_id in zip(boxes.xyxy.numpy(), track_ids, strict=True):
            x1, y1, x2, y2 = box.tolist()
            head = self._smooth_head(track_id, ((x1 + x2) // 2, y1))
            assignment = self._assign(track_id, visible_labels)
            people.append(TrackedPerson(track_id, head, assignment, (x1, y1, x2, y2)))
        return people

    def _assign(self, track_id: int, visible_labels: set[str]) -> Assignment:
        assignment = self.assignments.get(track_id)
        if assignment is None:
            pool_size = len(config.DONOR_LABELS)
            index = self._donor_cursor
            for offset in range(pool_size):
                candidate = (self._donor_cursor + offset) % pool_size
                if config.DONOR_LABELS[candidate] not in visible_labels:
                    index = candidate
                    break
            assignment = Assignment(
                nickname=config.DONOR_LABELS[index],
                color=config.BADGE_COLORS[index % len(config.BADGE_COLORS)],
                assigned_at=time(),
            )
            self.assignments[track_id] = assignment
            self._donor_cursor = (index + 1) % pool_size
        visible_labels.add(assignment.nickname)
        return assignment

    def _smooth_head(self, track_id: int, point: Point) -> Point:
        previous = self._heads.get(track_id)
        if previous is not None:
            old_x, old_y = previous.point
            x, y = point
            alpha = config.EMA_ALPHA
            point = (old_x + alpha * (x - old_x), old_y + alpha * (y - old_y))
        self._heads[track_id] = HeadPosition(point, self._frame_number)
        return point
