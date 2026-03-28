import time

import numpy as np
import pyautogui
from pynput.keyboard import Controller, Key

pyautogui.FAILSAFE = False   # disable corner-abort for gesture use
pyautogui.PAUSE = 0


class ActionMapper:
    def __init__(
        self,
        frame_width,
        frame_height,
        smoothing_alpha=0.4,
        action_cooldown=0.12,
        movement_margin=120,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_alpha = smoothing_alpha
        self.action_cooldown = action_cooldown
        self.movement_margin = movement_margin

        self.keyboard = Controller()
        self.last_action_time = {}
        self.scroll_anchor_y = None
        self.smoothed_x = None
        self.smoothed_y = None

    def execute(self, gesture, positions):
        if not positions:
            self.scroll_anchor_y = None
            return

        if gesture == "point":
            self._move_mouse(positions[8])
            self.scroll_anchor_y = None
            return

        if gesture == "pinch":
            self._scroll(positions[8][1])
            return

        self.scroll_anchor_y = None

        if gesture in ("none", "unknown", "fist", "out_of_zone"):
            return

        if not self._can_trigger(gesture):
            return

        actions = {
            "peace": self._left_click,
            "right_click": self._right_click,
            "open_hand": self._alt_tab,
            "thumbs_up": self._volume_up,
            "four_fingers": self._volume_down,
            "rock": self._mute_toggle,
        }
        action = actions.get(gesture)
        if action:
            action()

    def _move_mouse(self, tip):
        # Map index finger tip to screen coordinates with smoothing.
        screen_w, screen_h = pyautogui.size()

        x = np.interp(
            tip[0],
            [self.movement_margin, self.frame_width - self.movement_margin],
            [0, screen_w],
        )
        y = np.interp(
            tip[1],
            [self.movement_margin, self.frame_height - self.movement_margin],
            [0, screen_h],
        )

        x = float(np.clip(x, 0, screen_w - 1))
        y = float(np.clip(y, 0, screen_h - 1))

        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x, self.smoothed_y = x, y
        else:
            alpha = self.smoothing_alpha
            self.smoothed_x = alpha * x + (1.0 - alpha) * self.smoothed_x
            self.smoothed_y = alpha * y + (1.0 - alpha) * self.smoothed_y

        pyautogui.moveTo(self.smoothed_x, self.smoothed_y)

    def _scroll(self, current_y):
        if self.scroll_anchor_y is None:
            self.scroll_anchor_y = current_y
            return

        delta = current_y - self.scroll_anchor_y
        if abs(delta) < 8:
            return

        scroll_amount = int(np.clip(-delta * 1.5, -60, 60))
        pyautogui.scroll(scroll_amount)
        self.scroll_anchor_y = current_y

    def _left_click(self):
        pyautogui.click()

    def _right_click(self):
        pyautogui.rightClick()

    def _alt_tab(self):
        self.keyboard.press(Key.alt)
        self.keyboard.press(Key.tab)
        self.keyboard.release(Key.tab)
        self.keyboard.release(Key.alt)

    @staticmethod
    def _volume_up():
        pyautogui.press("volumeup")

    @staticmethod
    def _volume_down():
        pyautogui.press("volumedown")

    @staticmethod
    def _mute_toggle():
        pyautogui.press("volumemute")

    def _can_trigger(self, gesture):
        now = time.time()
        if now - self.last_action_time.get(gesture, 0.0) < self.action_cooldown:
            return False
        self.last_action_time[gesture] = now
        return True
