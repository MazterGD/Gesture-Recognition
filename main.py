import cv2
from hand_detector import HandDetector
from gesture_engine import GestureEngine
from action_mapper import ActionMapper
from face_tracker import FaceTracker
from monitor_mapper import MonitorMapper
from gaze_cursor import GazeCursor
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
    max_hands=2,
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
    action_cooldown=1.0,
)

face_tracker = FaceTracker()
monitor_mapper = MonitorMapper()
gaze_cursor = GazeCursor(monitor_mapper)
gaze_calibrated = False
gaze_tracking_enabled = True

fps_counter = FPSCounter()
debug_mode = False
start_time = time.time()
stats = {
    "index_up": 0,
    "index_down": 0,
    "peace": 0,
    "pinch_zoom": 0,
    "open_hand": 0,
    "thumbs_up": 0,
    "thumbs_down": 0,
    "thumb_pinky": 0,
    "three_fingers": 0,
    "four_fingers": 0,
    "play_pause": 0,
    "close_window": 0,
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
        face_data = face_tracker.process(frame)

        gesture = "none"
        fingers_up = []
        confidence = 0.0
        left_modifier_active = False

        if hands:
            right_hand = None
            left_hand = None

            for hand in hands:
                hand_label = hand["handedness"]
                if hand_label == "Right" and right_hand is None:
                    right_hand = hand
                elif hand_label == "Left" and left_hand is None:
                    left_hand = hand

            control_hand = right_hand if right_hand is not None else hands[0]
            positions = control_hand["landmarks"]
            handedness = control_hand["handedness"]
            confidence = control_hand["confidence"]

            if left_hand is not None:
                left_fingers = engine.get_fingers_up(left_hand["landmarks"], left_hand["handedness"])
                left_modifier_active = left_fingers == [False, False, True, True, True]

            if confidence >= CONFIG["HAND_DETECTION_CONF"] and len(positions) > 8:
                index_tip = positions[8]
                if inside_detection_zone(index_tip, zone):
                    gesture, fingers_up = engine.classify(positions, handedness)
                    mapper.execute(gesture, positions, handedness, left_modifier_active)
                else:
                    gesture = "out_of_zone"

        if gesture in stats:
            stats[gesture] += 1

        if face_data:
            gaze_x, gaze_y, yaw, pitch = face_data

            if gaze_calibrated and gaze_tracking_enabled:
                gaze_cursor.update(gaze_x, gaze_y, yaw, pitch)

            if debug_mode:
                cv2.putText(
                    frame,
                    f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})",
                    (20, 172),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Head yaw: {yaw:.1f}  pitch: {pitch:.1f}",
                    (20, 204),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 255, 0),
                    2,
                )

        gaze_status = "ON" if gaze_tracking_enabled else "PAUSED"
        cv2.putText(
            frame,
            f"Eye tracking: {gaze_status}",
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0) if gaze_tracking_enabled else (80, 80, 255),
            2,
        )

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
        if key == ord("c") and face_data:
            _, _, yaw, pitch = face_data
            gaze_cursor.calibrate(yaw, pitch)
            gaze_calibrated = True
        if key == ord("g"):
            gaze_tracking_enabled = not gaze_tracking_enabled
            print(
                "Eye tracking resumed"
                if gaze_tracking_enabled
                else "Eye tracking paused"
            )
finally:
    cap.release()
    detector.close()
    face_tracker.close()
    cv2.destroyAllWindows()
