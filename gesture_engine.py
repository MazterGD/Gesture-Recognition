import math


class GestureEngine:
    # Finger tip and base landmark IDs
    FINGER_TIPS  = [4, 8, 12, 16, 20]
    FINGER_BASES = [2, 6, 10, 14, 18]   # MCP joints

    def __init__(self, pinch_threshold=45, spread_threshold=55):
        self.pinch_threshold = pinch_threshold
        self.spread_threshold = spread_threshold

    def get_fingers_up(self, positions, handedness="Right"):
        """Returns list of booleans: [thumb, index, middle, ring, pinky]"""
        if not positions or len(positions) < 21:
            return []

        fingers = []

        # Thumb orientation flips between left and right hands.
        if handedness == "Right":
            fingers.append(positions[4][0] < positions[3][0])
        else:
            fingers.append(positions[4][0] > positions[3][0])

        # Other 4 fingers: compare y-axis (tip above base = up)
        for tip, base in zip(self.FINGER_TIPS[1:], self.FINGER_BASES[1:]):
            fingers.append(positions[tip][1] < positions[base][1])
        return fingers

    def classify(self, positions, handedness="Right"):
        fingers = self.get_fingers_up(positions, handedness)
        if not fingers:
            return "none", []

        pinch = self._is_pinch(positions)
        spread = self._distance(positions[8], positions[12])

        if pinch and not fingers[2] and not fingers[3] and not fingers[4]:
            return "pinch", fingers

        if fingers == [False, False, False, False, False]:
            return "fist", fingers
        if fingers == [False, True, False, False, False]:
            return "point", fingers
        if fingers == [False, True, True, False, False]:
            if spread >= self.spread_threshold:
                return "right_click", fingers
            return "peace", fingers
        if fingers == [True, True, True, True, True]:
            return "open_hand", fingers
        if fingers == [True, False, False, False, False]:
            return "thumbs_up", fingers
        if fingers == [False, True, True, True, True]:
            return "four_fingers", fingers
        if fingers == [True, False, False, False, True]:
            return "rock", fingers
        return "unknown", fingers

    @staticmethod
    def _distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _is_pinch(self, positions):
        return self._distance(positions[4], positions[8]) <= self.pinch_threshold
