import cv2

# Abre la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada, puedes cambiarlo si tienes múltiples cámaras
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()
frame_numero = 0

while True:
    # Lee un frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar el frame.")
        break
    
    frame_numero += 1

    #faces = app.get(frame)
    #rimg = app.draw_on(frame, faces)
    #cv2.imwrite(f"./{frame_numero}_output.jpg"  , rimg)

    # Muestra el número del frame en la esquina superior izquierda
    cv2.putText(rimg, f'Frame: {frame_numero}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Muestra el frame en una ventana
    cv2.imshow('Frame de la Cámara', rimg)
    # Sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') or frame_numero == 100:
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()