
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
import time
import serial

# ser = serial.Serial("COM19", 9600) #serial данные arduino (сом порт и скорость)

ser = serial.Serial('COM19',9600)


# Глобальные переменные управления моторами
a = 0
b = 0

last_detection_time = 0
last_detected_pts = None
center_coordinates = None
qr_distance = None

def detect_and_draw_qr(frame, target_data):
    global last_detection_time, last_detected_pts, center_coordinates, qr_distance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    decoded_objects = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
    min_distance = None
    current_time = time.time()
    detected = False

    for obj in decoded_objects:
        data = obj.data.decode('utf-8')
        if data == target_data:
            points = obj.polygon
            if len(points) == 4:
                pts = np.array(points, dtype=np.int32)
                last_detected_pts = pts
                last_detection_time = current_time
                detected = True

                center_x = int(sum(pt[0] for pt in pts) / 4)
                center_y = int(sum(pt[1] for pt in pts) / 4)
                center_coordinates = (center_x, center_y)

                width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
                focal_length = 600
                real_qr_width = 8.5
                distance = (real_qr_width * focal_length) / width
                min_distance = distance if min_distance is None else min(min_distance, distance)
                qr_distance = min_distance

    if not detected and last_detected_pts is not None and (current_time - last_detection_time) < 0.1:
        cv2.polylines(frame, [last_detected_pts], isClosed=True, color=(0, 255, 0), thickness=3)
        if center_coordinates:
            cv2.circle(frame, center_coordinates, 5, (0, 255, 0), -1)
    elif detected:
        cv2.polylines(frame, [last_detected_pts], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.circle(frame, center_coordinates, 5, (0, 255, 0), -1)

def control_logic(frame_width):
    global a, b, center_coordinates, qr_distance

    if center_coordinates is None or qr_distance is None:
        a = 127
        b = 127
    else:
        # Параметры зоны и допустимой дистанции
        rect_left = frame_width // 2 - 200
        rect_right = frame_width // 2 + 200
        center_x = frame_width // 2
        x, _ = center_coordinates

        # Дистанционный offset: -1 (слишком близко), 0 (норма), 1 (далеко)
        offset_d = np.clip((qr_distance - 45) / 15, -1.0, 1.0)
        speed = int(offset_d * 100)

        # Смещение по горизонтали, если QR вне зоны
        offset_x = 0
        if x < rect_left or x > rect_right:
            offset_x = np.clip((x - center_x) / (frame_width // 2), -1.0, 1.0)

        turn = int(offset_x * 100)

        # Итоговые значения моторов
        left = 127 + speed - turn
        right = 127 + speed + turn

        a = int(np.clip(left, 0, 255))
        b = int(np.clip(right, 0, 255))

    # Отправка значений каждый кадр
    message = f"{a},{b}\n"
    ser.write(message.encode())
def draw_overlay(frame):
    frame_height, frame_width = frame.shape[:2]

    # Область контроля по положению QR
    rect_left = frame_width // 2 - 200
    rect_right = frame_width // 2 + 200
    rect_top = 0
    rect_bottom = frame_height

    # Рисуем прямоугольную область
    cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (255, 0, 0), 2)

    # Пишем координаты и расстояние
    if center_coordinates:
        cv2.putText(frame, f"Position: {center_coordinates}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if qr_distance:
        cv2.putText(frame, f"Distance: {qr_distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Вывод управляющих значений
    cv2.putText(frame, f"a = {a}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
    cv2.putText(frame, f"b = {b}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

target_qr = "1"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_draw_qr(frame, target_qr)
    control_logic(frame.shape[1])
    draw_overlay(frame)

    cv2.imshow('Decoder', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()