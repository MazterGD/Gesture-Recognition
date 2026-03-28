import cv2
from hand_detector import HandDetector
from gesture_engine import GestureEngine
from action_mapper import ActionMapper
import time
from utils import CONFIG, FPSCounter, draw_status, get_detection_zone, inside_detection_zone


def print_stats(stats, uptime):
    print("\n--- Gesture Stats ---")
    print(f"Uptime: {uptime:.1f}s")
    for gesture, count in sorted(stats.items(), key=lambda item: item[1], reverse=True):
        if count > 0:
            print(f"{gesture:>12}: {count}")
    print("---------------------")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["CAMERA_WIDTH"])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["CAMERA_HEIGHT"])
cap.set(cv2.CAP_PROP_FPS, CONFIG["CAMERA_FPS"])

detector = HandDetector(
    detection_conf=CONFIG["HAND_DETECTION_CONF"],
    tracking_conf=CONFIG["HAND_TRACKING_CONF"],
)
engine = GestureEngine(
    pinch_threshold=CONFIG["PINCH_THRESHOLD"],
    spread_threshold=CONFIG["SPREAD_THRESHOLD"],
)
mapper = ActionMapper(
    frame_width=CONFIG["CAMERA_WIDTH"],
    frame_height=CONFIG["CAMERA_HEIGHT"],
    smoothing_alpha=CONFIG["SMOOTHING_ALPHA"],
    action_cooldown=CONFIG["ACTION_COOLDOWN"],
)

fps_counter = FPSCounter()
debug_mode = False
start_time = time.time()
stats = {
    "point": 0,
    "peace": 0,
    "right_click": 0,
    "pinch": 0,
    "open_hand": 0,
    "thumbs_up": 0,
    "four_fingers": 0,
    "rock": 0,
    "fist": 0,
}

zone = get_detection_zone(
    CONFIG["CAMERA_WIDTH"],
    CONFIG["CAMERA_HEIGHT"],
    CONFIG["ZONE_MARGIN_X"],
    CONFIG["ZONE_MARGIN_Y"],
)

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, hands = detector.find_hands(frame, draw=True)

        gesture = "none"
        fingers_up = []
        confidence = 0.0

        if hands:
            primary_hand = hands[0]
            positions = primary_hand["landmarks"]
            handedness = primary_hand["handedness"]
            confidence = primary_hand["confidence"]

            if confidence >= CONFIG["HAND_DETECTION_CONF"] and len(positions) > 8:
                index_tip = positions[8]
                if inside_detection_zone(index_tip, zone):
                    gesture, fingers_up = engine.classify(positions, handedness)
                    mapper.execute(gesture, positions)
                else:
                    gesture = "out_of_zone"

        if gesture in stats:
            stats[gesture] += 1

        fps = fps_counter.update()
        draw_status(frame, gesture, fps, confidence, fingers_up, zone, debug=debug_mode)

        cv2.imshow("Gesture Control", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("d"):
            debug_mode = not debug_mode
        if key == ord("s"):
            print_stats(stats, time.time() - start_time)
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
