from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

import config
from tracker import Color, Frame, TrackedPerson


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (22, 18, 28)
BADGE_SPACING = 4
BADGE_FONT = cv2.FONT_HERSHEY_DUPLEX
BADGE_SURFACE = (30, 23, 19)
BADGE_BORDER = (104, 80, 68)
AMOUNT_TEXT = (26, 22, 18)


@dataclass(frozen=True, slots=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    def overlaps(self, other: "Rect") -> bool:
        return (
            self.left < other.left + other.width + BADGE_SPACING
            and self.left + self.width + BADGE_SPACING > other.left
            and self.top < other.top + other.height + BADGE_SPACING
            and self.top + self.height + BADGE_SPACING > other.top
        )


@dataclass(frozen=True, slots=True)
class BadgeSprite:
    image: Frame
    opacity: NDArray[np.float32]
    transparency: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BadgePlacement:
    anchor: Rect
    bounds: Rect


class OverlayRenderer:
    def __init__(self, frame_height: int) -> None:
        self._scale = min(1.4, max(0.45, frame_height / 1800))
        self._thickness = max(1, round(self._scale * 1.6))
        self._padding = max(6, round(self._scale * 13))
        self._badges: dict[tuple[str, Color], BadgeSprite] = {}
        self._placements: dict[int, BadgePlacement] = {}

    def draw(
        self, frame: Frame, people: list[TrackedPerson], fps: float | None, device: str
    ) -> None:
        speed = f"{fps:5.1f}" if fps is not None else "  -- "
        caption = f"FPS {speed}   NPC {len(people)}   {device}"
        width, height = self._text_size(caption)
        left = top = 12
        pad = self._padding
        occupied = [Rect(left, top, width + 2 * pad, height + 2 * pad)]
        badges: list[tuple[TrackedPerson, Rect]] = []
        placements: dict[int, BadgePlacement] = {}
        # Stable ordering keeps the same track in front when two badges overlap.
        for person in sorted(people, key=lambda person: person.track_id):
            placement = self._place_badge(frame, person, occupied)
            if placement is not None:
                placements[person.track_id] = placement
                badges.append((person, placement.bounds))
                occupied.append(placement.bounds)
        self._placements = placements
        for person, bounds in badges:
            self._draw_leader(frame, person, bounds)
        for person, bounds in badges:
            self._draw_badge(frame, person, bounds)

        self._shade(frame, left, top, width + 2 * pad, height + 2 * pad)
        self._text(frame, caption, (left + pad, top + pad + height), TEXT_COLOR)

    def _place_badge(
        self, frame: Frame, person: TrackedPerson, occupied: list[Rect]
    ) -> BadgePlacement | None:
        sprite = self._badge_sprite(person)
        height, width = sprite.image.shape[:2]
        frame_height, frame_width = frame.shape[:2]
        if width > frame_width or height > frame_height:
            return None

        x, y = map(round, person.head)
        max_left = frame_width - width
        max_top = frame_height - height
        anchor = Rect(
            min(max_left, max(0, x - width // 2)),
            min(max_top, max(0, y - config.BADGE_HEAD_GAP - height)),
            width,
            height,
        )
        preferred = self._find_open_slot(anchor, max_left, occupied)
        previous = self._placements.get(person.track_id)
        bounds = preferred
        if previous is not None and previous.bounds != previous.anchor:
            # Follow the head at the displaced offset. Return to the anchor in
            # small steps as space opens around the badge.
            projected = Rect(
                min(max_left, max(0, anchor.left + previous.bounds.left - previous.anchor.left)),
                min(max_top, max(0, anchor.top + previous.bounds.top - previous.anchor.top)),
                width,
                height,
            )
            if not any(projected.overlaps(rect) for rect in occupied):
                bounds = projected
                if preferred is not None:
                    step = max(2, height // 8)
                    dx = max(-step, min(step, preferred.left - projected.left))
                    dy = max(-step, min(step, preferred.top - projected.top))
                    candidate = Rect(projected.left + dx, projected.top + dy, width, height)
                    if not any(candidate.overlaps(rect) for rect in occupied):
                        bounds = candidate
        return BadgePlacement(anchor, bounds) if bounds is not None else None

    @staticmethod
    def _find_open_slot(anchor: Rect, max_left: int, occupied: list[Rect]) -> Rect | None:
        width, height = anchor.width, anchor.height
        for offset in (0, -(width + BADGE_SPACING), width + BADGE_SPACING):
            left = max(0, min(anchor.left + offset, max_left))
            top = anchor.top
            while top >= 0:
                candidate = Rect(left, top, width, height)
                blockers = [rect.top for rect in occupied if candidate.overlaps(rect)]
                if not blockers:
                    return candidate
                top = min(blockers) - height - BADGE_SPACING
        return None

    @staticmethod
    def _draw_leader(frame: Frame, person: TrackedPerson, bounds: Rect) -> None:
        x, y = map(round, person.head)
        center = bounds.left + bounds.width // 2
        bottom = bounds.top + bounds.height
        if center == x and bottom == y - config.BADGE_HEAD_GAP:
            return
        end = (x, y - 2)
        start = (center, bottom - 1) if end[1] >= bottom else (center, bounds.top)
        cv2.line(frame, start, end, BADGE_SURFACE, 3, cv2.LINE_AA)
        cv2.line(frame, start, end, person.assignment.color, 1, cv2.LINE_AA)

    def _draw_badge(self, frame: Frame, person: TrackedPerson, bounds: Rect) -> None:
        sprite = self._badge_sprite(person)
        patch = frame[
            bounds.top : bounds.top + bounds.height,
            bounds.left : bounds.left + bounds.width,
        ]
        cv2.blendLinear(sprite.image, patch, sprite.opacity, sprite.transparency, dst=patch)

    def _badge_sprite(self, person: TrackedPerson) -> BadgeSprite:
        label, accent = person.assignment.nickname, person.assignment.color
        key = (label, accent)
        sprite = self._badges.get(key)
        if sprite is not None:
            return sprite

        name, _, amount = label.partition(" • ")
        name_scale = self._scale * 1.08
        amount_scale = self._scale * 0.87
        stroke = self._thickness
        name_width, name_height = cv2.getTextSize(name, BADGE_FONT, name_scale, stroke)[0]
        amount_width, amount_height = cv2.getTextSize(amount, BADGE_FONT, amount_scale, stroke)[0]
        pad = self._padding
        accent_width = max(2, round(self._scale * 4))
        name_left = pad + accent_width + max(4, pad // 2)
        capsule_height = amount_height + 2 * max(4, pad - 2)
        height = max(name_height + 2 * pad, capsule_height + 2 * max(3, pad // 2))
        width = name_left + name_width + pad
        if amount:
            width += amount_width + 2 * pad

        image = np.full((height, width, 3), BADGE_BORDER, dtype=np.uint8)
        radius = max(5, round(self._scale * 15))
        self._rounded_fill(image, Rect(1, 1, width - 2, height - 2), radius - 1, BADGE_SURFACE)
        accent_height = max(8, height // 2)
        self._rounded_fill(
            image, Rect(pad, (height - accent_height) // 2, accent_width, accent_height), 1, accent
        )
        cv2.putText(
            image, name, (name_left, (height + name_height) // 2 - 1),
            BADGE_FONT, name_scale, TEXT_COLOR, stroke, cv2.LINE_AA,
        )
        if amount:
            capsule_left = name_left + name_width + pad // 2
            capsule_top = (height - capsule_height) // 2
            capsule_width = amount_width + 2 * pad
            capsule = Rect(capsule_left, capsule_top, capsule_width, capsule_height)
            self._rounded_fill(image, capsule, (capsule_height - 1) // 2, accent)
            cv2.putText(
                image, amount, (capsule_left + pad, (height + amount_height) // 2 - 1),
                BADGE_FONT, amount_scale, AMOUNT_TEXT, stroke, cv2.LINE_AA,
            )

        mask = np.zeros((height, width), dtype=np.uint8)
        self._rounded_fill(mask, Rect(0, 0, width, height), radius, 240)
        opacity = mask.astype(np.float32) / 255.0
        sprite = BadgeSprite(image, opacity, 1.0 - opacity)
        self._badges[key] = sprite
        return sprite

    @staticmethod
    def _rounded_fill(
        image: NDArray[np.uint8], bounds: Rect, radius: int, color: Color | int
    ) -> None:
        l, t = bounds.left, bounds.top
        r, b = l + bounds.width - 1, t + bounds.height - 1
        radius = min(radius, (bounds.width - 1) // 2, (bounds.height - 1) // 2)
        cv2.rectangle(image, (l + radius, t), (r - radius, b), color, -1)
        cv2.rectangle(image, (l, t + radius), (r, b - radius), color, -1)
        for x in (l + radius, r - radius):
            for y in (t + radius, b - radius):
                cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)

    @staticmethod
    def _shade(frame: Frame, left: int, top: int, width: int, height: int) -> None:
        right, bottom = left + width, top + height
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(frame.shape[1], right), min(frame.shape[0], bottom)
        if x1 >= x2 or y1 >= y2:
            return

        patch = frame[y1:y2, x1:x2]
        overlay = patch.copy()
        OverlayRenderer._rounded_fill(
            overlay, Rect(left - x1, top - y1, width, height), 7, PANEL_COLOR
        )
        cv2.addWeighted(overlay, 0.76, patch, 0.24, 0, dst=patch)

    def _text_size(self, text: str) -> tuple[int, int]:
        return cv2.getTextSize(text, FONT, self._scale, self._thickness)[0]

    def _text(self, frame: Frame, text: str, position: tuple[int, int], color: Color) -> None:
        cv2.putText(
            frame, text, position, FONT, self._scale, color, self._thickness, cv2.LINE_AA
        )
