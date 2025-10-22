import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Función para determinar la letra según la posición de los dedos
def reconocer_letra(hand_landmarks, frame):
    h, w, _ = frame.shape  # Tamaño de la imagen
    
    # Obtener coordenadas de los puntos clave en píxeles
    dedos = [(int(hand_landmarks.landmark[i].x * w), int(hand_landmarks.landmark[i].y * h)) for i in range(21)]
    
    # Obtener posiciones clave (puntas de los dedos)
    pulgar, indice, medio, anular, meñique = dedos[4], dedos[8], dedos[12], dedos[16], dedos[20]

    # Mostrar los números de los landmarks en la imagen
    for i, (x, y) in enumerate(dedos):
        cv2.circle(frame, (x, y), 5, (233, 23, 0), -1)  # Puntos verdes
        cv2.putText(frame, str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Calcular distancias en píxeles
    distancia_pulgar_indice = np.linalg.norm(np.array(pulgar) - np.array(indice))
    distancia_indice_medio = np.linalg.norm(np.array(indice) - np.array(medio))



    centro_x = (pulgar[0] + indice[0]) // 2
    centro_y = (pulgar[1] + indice[1]) // 2
    
    tamaño_cuadrado = int(distancia_pulgar_indice)  # 80% de la distancia
    
    x1 = centro_x - tamaño_cuadrado // 2
    y1 = centro_y - tamaño_cuadrado // 2
    x2 = centro_x + tamaño_cuadrado // 2
    y2 = centro_y + tamaño_cuadrado // 2
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 234, 65), 3)
    
    
 
# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)

    # Convertir a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(frame_rgb)

    # Dibujar puntos de la mano y reconocer letras
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Identificar la letra
            letra_detectada = reconocer_letra(hand_landmarks, frame)

            # Mostrar la letra en pantalla
            cv2.putText(frame, f"Letra: {letra_detectada}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video
    cv2.imshow("Reconocimiento de Letras con Cuadrado", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()