import time

import cv2


CONFIG = {
	"CAMERA_WIDTH": 1280,
	"CAMERA_HEIGHT": 720,
	"CAMERA_FPS": 30,
	"HAND_DETECTION_CONF": 0.75,
	"HAND_TRACKING_CONF": 0.75,
	"SMOOTHING_ALPHA": 0.4,
	"ACTION_COOLDOWN": 0.12,
	"PINCH_THRESHOLD": 45,
	"SPREAD_THRESHOLD": 55,
	"ZONE_MARGIN_X": 140,
	"ZONE_MARGIN_Y": 90,
}


class FPSCounter:
	def __init__(self):
		self.prev_t = time.time()

	def update(self):
		curr_t = time.time()
		fps = 1.0 / max(curr_t - self.prev_t, 1e-6)
		self.prev_t = curr_t
		return fps


def get_detection_zone(frame_width, frame_height, margin_x, margin_y):
	return (margin_x, margin_y, frame_width - margin_x, frame_height - margin_y)


def inside_detection_zone(point, zone):
	x1, y1, x2, y2 = zone
	return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def draw_status(frame, gesture, fps, confidence, fingers_up, zone, debug=False):
	x1, y1, x2, y2 = zone
	cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 180, 255), 2)

	cv2.putText(
		frame,
		f"Gesture: {gesture}",
		(20, 36),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(0, 255, 100),
		2,
	)
	cv2.putText(
		frame,
		f"FPS: {fps:.0f}",
		(20, 70),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(255, 255, 255),
		2,
	)
	cv2.putText(
		frame,
		f"Hand conf: {confidence:.2f}",
		(20, 104),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.7,
		(255, 255, 255),
		2,
	)

	if debug:
		cv2.putText(
			frame,
			f"Fingers: {fingers_up}",
			(20, 138),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(230, 230, 90),
			2,
		)
		cv2.putText(
			frame,
			"Keys: q quit | d debug | s stats | c calibrate | g gaze on/off",
			(20, frame.shape[0] - 20),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(220, 220, 220),
			2,
		)
