import cv2
import numpy as np
import threading
import pygame
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")

alarm_on = False

def tocar_alarme():
    global alarm_on
    while alarm_on:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

def iniciar_alarme():
    global alarm_on
    if not alarm_on:
        alarm_on = True
        threading.Thread(target=tocar_alarme, daemon=True).start()

def parar_alarme():
    global alarm_on
    alarm_on = False
    pygame.mixer.music.stop()

def calcular_ear(pontos):
    p1, p2, p3, p4, p5, p6 = pontos
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))


def gerar_frames():
    cap = cv2.VideoCapture(0)

    EAR_LIMITE = 0.25
    TEMPO_LIMITE = 2
    FPS = 30

    frames_fechados = 0
    max_frames = TEMPO_LIMITE * FPS

    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            face = result.face_landmarks[0]

            olho_esq_ids = [33, 160, 158, 133, 153, 144]
            olho_dir_ids = [362, 385, 387, 263, 373, 380]

            def get_pontos(ids):
                return np.array([[face[i].x * w, face[i].y * h] for i in ids])

            ear = (calcular_ear(get_pontos(olho_esq_ids)) +
                   calcular_ear(get_pontos(olho_dir_ids))) / 2

            # Desenhar EAR na tela
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if ear < EAR_LIMITE:
                frames_fechados += 1
            else:
                frames_fechados = 0
                parar_alarme()

            if frames_fechados >= max_frames:
                iniciar_alarme()
                cv2.putText(frame, "ACORDE!", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        # Converter frame para stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')