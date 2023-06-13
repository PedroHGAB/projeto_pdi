import cv2
import mediapipe as mp
from math import dist

# Inicializa o módulo de rastreamento de mãos do Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Define o fator de zoom inicial
zoom_factor = 1.0

# Define os limites mínimo e máximo para o fator de zoom
min_zoom_factor = 1.0
max_zoom_factor = 3.0

# Inicializa a distância inicial como None
initial_distance = None

while True:
    # Lê o próximo quadro da câmera
    ret, frame = cap.read()

    if not ret:
        break

    # Inverte o quadro horizontalmente para corresponder ao movimento da mão
    frame = cv2.flip(frame, 1)

    # Converte o quadro para o formato RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa o quadro com o módulo de rastreamento de mãos
    results = hands.process(frame_rgb)

    # Verifica se há pelo menos uma mão detectada
    if results.multi_hand_landmarks:
        # Obtém as coordenadas dos pontos-chave da mão
        landmarks = results.multi_hand_landmarks[0].landmark

        # Obtém a posição dos dedos indicador e polegar
        thumb_tip = (int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]),
                     int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]))
        index_tip = (int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]),
                     int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]))

        # Calcula a distância entre os pontos indicador e polegar
        current_distance = dist(thumb_tip, index_tip)

        # Verifica se a distância inicial foi definida
        if initial_distance is not None:
            # Calcula o novo fator de zoom com base na mudança na distância
            zoom_factor *= current_distance / initial_distance

            # Limita o fator de zoom dentro dos limites mínimo e máximo
            zoom_factor = max(min_zoom_factor, min(max_zoom_factor, zoom_factor))

        # Atualiza a distância inicial para o próximo cálculo
        initial_distance = current_distance

    # Aplica o fator de zoom ao quadro
    zoomed_frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor)

    # Exibe o quadro com zoom
    cv2.imshow('Zoomed Frame', zoomed_frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a
