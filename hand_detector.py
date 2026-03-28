import cv2
from mediapipe.python.solutions import drawing_utils
from mediapipe.python.solutions import hands as mp_hands


class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.8, tracking_conf=0.8):
        self.mp_hands = mp_hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = drawing_utils
        self.results = None

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        hands_data = []
        hand_landmarks = getattr(self.results, "multi_hand_landmarks", None)

        if hand_landmarks and draw:
            for idx, hand_lm in enumerate(hand_landmarks):
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, list(self.mp_hands.HAND_CONNECTIONS)
                )
                lm_list = self._extract_landmarks(frame, hand_lm)
                handedness, confidence = self._extract_handedness(idx)
                hands_data.append(
                    {
                        "landmarks": lm_list,
                        "handedness": handedness,
                        "confidence": confidence,
                    }
                )
        elif hand_landmarks:
            for idx, hand_lm in enumerate(hand_landmarks):
                lm_list = self._extract_landmarks(frame, hand_lm)
                handedness, confidence = self._extract_handedness(idx)
                hands_data.append(
                    {
                        "landmarks": lm_list,
                        "handedness": handedness,
                        "confidence": confidence,
                    }
                )

        return frame, hands_data

    def get_landmark_positions(self, frame):
        """Returns list of (x, y) pixel positions for all 21 landmarks."""
        h, w, _ = frame.shape
        positions = []
        hand_landmarks = getattr(self.results, "multi_hand_landmarks", None)
        if hand_landmarks:
            for lm in hand_landmarks[0].landmark:
                positions.append((int(lm.x * w), int(lm.y * h)))
        return positions

    def close(self):
        self.hands.close()

    @staticmethod
    def _extract_landmarks(frame, hand_lm):
        h, w, _ = frame.shape
        return [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm.landmark]

    def _extract_handedness(self, idx):
        handedness = "Unknown"
        confidence = 0.0
        multi_handedness = getattr(self.results, "multi_handedness", None)
        if multi_handedness and idx < len(multi_handedness):
            classification = multi_handedness[idx].classification[0]
            handedness = classification.label
            confidence = float(classification.score)
        return handedness, confidence
