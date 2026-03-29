import time
from collections import deque

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
        action_cooldown=0.5,
        movement_margin=120,
        volume_cooldown_multiplier=1.0,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.smoothing_alpha = smoothing_alpha
        self.action_cooldown = action_cooldown
        self.volume_cooldown_multiplier = volume_cooldown_multiplier
        self.movement_margin = movement_margin

        self.keyboard = Controller()
        self.last_action_time = {}
        self.last_global_action_time = 0.0
        self.scroll_anchor_y = None
        self.smoothed_x = None
        self.smoothed_y = None

        self.current_gesture = "none"
        self.hold_counter = 0

        self.index_y_history = deque(maxlen=8)
        self.palm_x_history = deque(maxlen=8)
        self.pinch_distance_history = deque(maxlen=12)
        self.last_scroll_time = 0.0
        self.last_zoom_time = 0.0

    def execute(self, gesture, positions, handedness="Right", left_modifier_active=False):
        if not positions:
            self.scroll_anchor_y = None
            self.current_gesture = "none"
            self.hold_counter = 0
            self.index_y_history.clear()
            self.palm_x_history.clear()
            self.pinch_distance_history.clear()
            return

        if gesture == self.current_gesture:
            self.hold_counter += 1
        else:
            self.current_gesture = gesture
            self.hold_counter = 1

        self.index_y_history.append(positions[8][1])
        self.palm_x_history.append(positions[0][0])
        pinch_distance = float(np.hypot(positions[4][0] - positions[8][0], positions[4][1] - positions[8][1]))
        self.pinch_distance_history.append(pinch_distance)

        if gesture == "index_up":
            self._move_mouse(positions[8])
            self.scroll_anchor_y = None

            if self._is_click_flick() and self._can_trigger("left_click"):
                self._left_click()
            return

        if gesture == "pinch_zoom":
            if not left_modifier_active:
                return
            self._zoom_from_pinch_distance()
            return

        if gesture in ("none", "unknown", "fist", "out_of_zone"):
            self.scroll_anchor_y = None
            return

        if gesture == "peace":
            if self._peace_vertical_motion() and time.time() - self.last_scroll_time > 0.08:
                self._scroll(positions[8][1])
                self.last_scroll_time = time.time()
                return

            self.scroll_anchor_y = None
            if self.hold_counter >= 10 and self._can_trigger("right_click"):
                self._right_click()
            return

        self.scroll_anchor_y = None

        if gesture == "open_hand":
            if handedness == "Left":
                if self.hold_counter >= 10 and self._can_trigger("play_pause"):
                    self._play_pause()
                return

            swipe = self._horizontal_swipe()
            if swipe < -45 and self._can_trigger("desktop_next"):
                self._desktop_next()
            elif swipe > 45 and self._can_trigger("desktop_prev"):
                self._desktop_prev()
            return

        if gesture == "thumb_pinky" and self._can_trigger("mute_toggle"):
            self._mute_toggle()
            return

        if gesture == "three_fingers" and self._can_trigger("alt_tab"):
            self._alt_tab()
            return

        if gesture == "four_fingers" and self._can_trigger("ctrl_tab"):
            self._ctrl_tab()
            return

        if (
            gesture == "thumbs_up"
            and handedness == "Right"
            and left_modifier_active
            and self._can_trigger("thumbs_up")
        ):
            self._volume_up()
            return

        if (
            gesture == "thumbs_down"
            and handedness == "Right"
            and left_modifier_active
            and self._can_trigger("thumbs_down")
        ):
            self._volume_down()
            return

        if gesture == "index_down" and self._can_trigger("minimize"):
            self._minimize_window()
            return

        if gesture == "close_window" and self._can_trigger("close_window"):
            self._close_window()

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
    def _ctrl_tab():
        pyautogui.hotkey("ctrl", "tab")

    @staticmethod
    def _volume_up():
        pyautogui.press("volumeup")

    @staticmethod
    def _volume_down():
        pyautogui.press("volumedown")

    @staticmethod
    def _mute_toggle():
        pyautogui.press("volumemute")

    @staticmethod
    def _play_pause():
        pyautogui.press("playpause")

    @staticmethod
    def _desktop_next():
        pyautogui.hotkey("ctrl", "win", "right")

    @staticmethod
    def _desktop_prev():
        pyautogui.hotkey("ctrl", "win", "left")

    @staticmethod
    def _close_window():
        pyautogui.hotkey("alt", "f4")

    @staticmethod
    def _minimize_window():
        pyautogui.hotkey("win", "down")

    def _zoom_from_pinch_distance(self):
        if len(self.pinch_distance_history) < 6:
            return

        now = time.time()
        if now - self.last_zoom_time < 0.2:
            return

        recent = list(self.pinch_distance_history)[-6:]
        delta = recent[-1] - recent[0]
        if delta > 25:
            pyautogui.hotkey("ctrl", "=")
            self.last_zoom_time = now
        elif delta < -25:
            pyautogui.hotkey("ctrl", "-")
            self.last_zoom_time = now

    def _is_click_flick(self):
        if len(self.pinch_distance_history) < 8:
            return False

        recent = list(self.pinch_distance_history)[-8:]
        return min(recent) < 35 and recent[-1] > 60

    def _peace_vertical_motion(self):
        if len(self.index_y_history) < 4:
            return False
        return abs(self.index_y_history[-1] - self.index_y_history[0]) > 24

    def _horizontal_swipe(self):
        if len(self.palm_x_history) < 4:
            return 0.0
        return float(self.palm_x_history[-1] - self.palm_x_history[0])

    def _can_trigger(self, gesture):
        now = time.time()

        # Global debounce so users have a clear gap between consecutive gestures.
        if now - self.last_global_action_time < self.action_cooldown:
            return False

        cooldown = self.action_cooldown
        if gesture in ("thumbs_up", "four_fingers"):
            cooldown *= self.volume_cooldown_multiplier

        if now - self.last_action_time.get(gesture, 0.0) < cooldown:
            return False

        self.last_action_time[gesture] = now
        self.last_global_action_time = now
        return True
