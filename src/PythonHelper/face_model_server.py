import cv2
import mediapipe as mp
import sys

mp_face = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
)

cap = cv2.VideoCapture(0)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    bSuccess, img = cap.read()
    if not bSuccess:
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
    sys.stdout.flush()