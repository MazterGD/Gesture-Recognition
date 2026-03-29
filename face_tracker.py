import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh


class FaceTracker:
    # Nose, chin, eye corners, mouth corners.
    HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]

    MODEL_3D_POINTS = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, detection_conf=0.7, tracking_conf=0.7):
        self.mp_face = mp_face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame):
        """Returns (gaze_x, gaze_y, yaw, pitch) or None when no face is found."""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        face_landmarks = getattr(results, "multi_face_landmarks", None)

        if not face_landmarks:
            return None

        lm = face_landmarks[0].landmark
        gaze_x, gaze_y = self._get_iris_gaze(lm, w, h)
        yaw, pitch = self._get_head_pose(lm, w, h)
        return gaze_x, gaze_y, yaw, pitch

    def close(self):
        self.face_mesh.close()

    def _get_iris_gaze(self, lm, w, h):
        def iris_offset(iris_idx, left_corner, right_corner, top_corner, bottom_corner):
            iris = np.array([lm[iris_idx].x * w, lm[iris_idx].y * h])
            left = np.array([lm[left_corner].x * w, lm[left_corner].y * h])
            right = np.array([lm[right_corner].x * w, lm[right_corner].y * h])
            top = np.array([lm[top_corner].x * w, lm[top_corner].y * h])
            bottom = np.array([lm[bottom_corner].x * w, lm[bottom_corner].y * h])

            gx = (iris[0] - left[0]) / (right[0] - left[0] + 1e-6) * 2.0 - 1.0
            gy = (iris[1] - top[1]) / (bottom[1] - top[1] + 1e-6) * 2.0 - 1.0
            return gx, gy

        lx, ly = iris_offset(468, 33, 133, 159, 145)
        rx, ry = iris_offset(473, 362, 263, 386, 374)

        gaze_x = float(np.clip((lx + rx) / 2.0, -1.0, 1.0))
        gaze_y = float(np.clip((ly + ry) / 2.0, -1.0, 1.0))
        return gaze_x, gaze_y

    def _get_head_pose(self, lm, w, h):
        image_points = np.array(
            [[lm[i].x * w, lm[i].y * h] for i in self.HEAD_POSE_POINTS],
            dtype=np.float64,
        )

        focal = float(w)
        camera_matrix = np.array(
            [
                [focal, 0, w / 2.0],
                [0, focal, h / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        success, rvec, _ = cv2.solvePnP(
            self.MODEL_3D_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        pitch = float(np.degrees(np.arctan2(-rmat[2, 0], sy)))
        yaw = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
        return yaw, pitch
