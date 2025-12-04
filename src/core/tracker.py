# src/core/tracker.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class ObjectTracker:
    def __init__(self, config):
        self.tracker = DeepSort(
            max_age=config.TRACKING_MAX_AGE,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None
        )

    def update(self, detections, frame):
        if not detections:
            return self.tracker.update_tracks([])

        # Конвертация детекций для DeepSORT
        deepsort_detections = []
        for det in detections:
            bbox = det['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            deepsort_detections.append([
                bbox[0], bbox[1], width, height, det['confidence']
            ])

        # Обновление трекера
        tracks = self.tracker.update_tracks(
            deepsort_detections,
            frame=frame
        )

        return tracks


# src/core/face_reid.py
from deepface import DeepFace
import cv2
import os


class FaceRecognition:
    def __init__(self, config):
        self.config = config
        self.known_faces = self.load_known_faces()

    def load_known_faces(self):
        faces = {}
        for filename in os.listdir(self.config.FACES_DIR):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                face_id = filename.split('.')[0]
                img_path = os.path.join(self.config.FACES_DIR, filename)
                faces[face_id] = img_path
        return faces

    def extract_faces(self, frame, detections):
        faces_data = []
        for track in detections:
            if track.is_confirmed():
                bbox = track.to_tlbr()
                face_roi = frame[int(bbox[1]):int(bbox[3]),
                int(bbox[0]):int(bbox[2])]

                if face_roi.size > 0:
                    identity = self.identify_face(face_roi)
                    faces_data.append({
                        'track_id': track.track_id,
                        'bbox': bbox,
                        'identity': identity,
                        'face_image': face_roi
                    })
        return faces_data

    def identify_face(self, face_image):
        for face_id, face_path in self.known_faces.items():
            try:
                result = DeepFace.verify(
                    face_image,
                    face_path,
                    model_name='Facenet',
                    detector_backend='opencv',
                    enforce_detection=False
                )
                if result['verified']:
                    return face_id
            except:
                continue
        return "unknown"