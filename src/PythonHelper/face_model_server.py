import struct
import cv2
import mediapipe as mp
import sys
import numpy as np

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
)

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def get_img_from_cpp():
    len = sys.stdin.buffer.read(4)
    if not len:
        return None
    
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(sys.stdin.buffer.read(struct.unpack('I', len)[0]), dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img

while True:
    img = get_img_from_cpp()
    if img is None:
        break

    result = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        continue

    landmarks = result.multi_face_landmarks[0].landmark
        
    components = []
    
    for pt in landmarks:
        components.append(str(int(pt.x * w)))
        components.append(str(int(pt.y * h)))

    sys.stdout.write(",".join(components) + "\n")
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        break