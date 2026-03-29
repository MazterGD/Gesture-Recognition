import numpy as np
import pyautogui


class GazeCursor:
    def __init__(
        self,
        monitor_mapper,
        head_weight=0.7,
        gaze_weight=0.3,
        alpha=0.15,
        yaw_range=35.0,
        pitch_range=20.0,
        head_deadzone=0.03,
        gaze_deadzone=0.05,
        jitter_threshold_pixels=6,
        invert_x=True,
    ):
        self.mapper = monitor_mapper
        self.head_weight = head_weight
        self.gaze_weight = gaze_weight
        self.alpha = alpha

        self.smooth_x = 0.0
        self.smooth_y = 0.0

        self.head_center_yaw = 0.0
        self.head_center_pitch = 0.0

        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.head_deadzone = head_deadzone
        self.gaze_deadzone = gaze_deadzone
        self.jitter_threshold_pixels = jitter_threshold_pixels
        self.invert_x = invert_x
        self.last_px = None
        self.last_py = None

    def calibrate(self, yaw, pitch):
        self.head_center_yaw = yaw
        self.head_center_pitch = pitch
        print(f"Calibrated center: yaw={yaw:.1f}, pitch={pitch:.1f}")

    def update(self, gaze_x, gaze_y, yaw, pitch):
        rel_yaw = (yaw - self.head_center_yaw) / max(self.yaw_range, 1e-6)
        rel_pitch = (pitch - self.head_center_pitch) / max(self.pitch_range, 1e-6)

        head_x = float(np.clip(rel_yaw, -1.0, 1.0))
        head_y = float(np.clip(rel_pitch, -1.0, 1.0))

        if abs(head_x) < self.head_deadzone:
            head_x = 0.0
        if abs(head_y) < self.head_deadzone:
            head_y = 0.0
        if abs(gaze_x) < self.gaze_deadzone:
            gaze_x = 0.0
        if abs(gaze_y) < self.gaze_deadzone:
            gaze_y = 0.0

        fused_x = self.head_weight * head_x + self.gaze_weight * gaze_x
        fused_y = self.head_weight * head_y + self.gaze_weight * gaze_y

        if self.invert_x:
            fused_x *= -1.0

        fused_x = float(np.clip(fused_x, -1.0, 1.0))
        fused_y = float(np.clip(fused_y, -1.0, 1.0))

        self.smooth_x = self.alpha * fused_x + (1.0 - self.alpha) * self.smooth_x
        self.smooth_y = self.alpha * fused_y + (1.0 - self.alpha) * self.smooth_y

        px, py = self.mapper.vector_to_screen_coords(self.smooth_x, self.smooth_y)

        if self.last_px is not None and self.last_py is not None:
            if abs(px - self.last_px) + abs(py - self.last_py) < self.jitter_threshold_pixels:
                return

        self.last_px = px
        self.last_py = py
        pyautogui.moveTo(px, py)
