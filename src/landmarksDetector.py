import cv2, mediapipe as mp, numpy as np

face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def run_model_on(bgr: np.ndarray) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
        
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = face.process(rgb)
    if not result.multi_face_landmarks:
        return np.empty((0, 2), dtype=np.float32)
    
    h, w = bgr.shape[:2]

    landmarks = result.multi_face_landmarks[0].landmark
    return np.asarray([[p.x * w, p.y * h] for p in landmarks], dtype=np.float32)