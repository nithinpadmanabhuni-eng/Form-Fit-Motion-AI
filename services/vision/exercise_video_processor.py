import os
import time
import cv2
import av
import numpy as np
import mediapipe as mp
import threading
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS

# Pose detection runs on a downscaled copy of the frame (landmarks are
# normalized, so they still map correctly onto the full-size image).
DETECT_WIDTH = 480


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        models_dir = os.path.join(os.getcwd(), "ml_models")
        lite_path = os.path.join(models_dir, "pose_landmarker_lite.task")
        full_path = os.path.join(models_dir, "pose_landmarker_full.task")

        # Prefer the lite model (much faster); fall back to full if not present
        model_path = lite_path if os.path.exists(lite_path) else full_path
        base_option = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        self._last_timestamp_ms = 0
    
    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()
        
    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type

    def _next_timestamp_ms(self):
        # Real clock time, strictly increasing (required by MediaPipe VIDEO mode)
        now_ms = int(time.monotonic() * 1000)

        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = now_ms

        return now_ms
        
    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.5 and p2.visibility > 0.5:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    4
                )
        
        for lm in landmarks:
            if lm.visibility > 0.5:
                cv2.circle(
                    img, 
                    (int(lm.x * w), int(lm.y * h)),
                    5,
                    (255, 0, 0),
                    -1
                )
            
    def _draw_no_pose_warnings(self, img):
        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_overlays(self, img, metrics, ex_type):
        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)
        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)
        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)
        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)
        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)


    def _draw_squats_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"DEPTH: {metrics['depth_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    
    def _draw_pushup_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BODY: {metrics['body_alignment']} | HIP: {metrics['hip_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_curl_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"SWING: {metrics['swing_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_press_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"EXT: {metrics['extension_status']} | BACK: {metrics['back_arch_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_lunge_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BALANCE: {metrics['balance_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _mark_no_pose(self):
        with self._lock:
            if self._latest_metrics is not None:
                self._latest_metrics["pose_detected"] = False
            else:
                self._latest_metrics = {"pose_detected": False}

    def recv(self, frame):
        image = np.ascontiguousarray(
            cv2.flip(frame.to_ndarray(format="bgr24"), 1),
            dtype=np.uint8
        )

        # Any exception inside recv() can stall or kill the video stream,
        # so never let one escape: just show the raw frame instead.
        try:
            h, w = image.shape[:2]

            if w > DETECT_WIDTH:
                small = cv2.resize(image, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)))
            else:
                small = image

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            )

            result = self._landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                self._draw_skeleton(image, landmarks)

                ex_type = self.get_exercise()
                detector = self._detectors.get(ex_type)

                if detector:
                    metrics = detector.process(landmarks)
                    metrics["pose_detected"] = True

                    self._draw_overlays(image, metrics, ex_type)
                    self.set_latest_metrics(metrics)
            else:
                self._draw_no_pose_warnings(image)
                self._mark_no_pose()

        except Exception as e:
            print(f"[VideoProcessor] frame error: {e}")

        return av.VideoFrame.from_ndarray(image, format="bgr24")