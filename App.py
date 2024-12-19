import streamlit as st
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import math

model = YOLO('yolov8x.pt')
model.overrides['conf'] = 0.3

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {obj[4]: self.center_points[obj[4]] for obj in objects_bbs_ids}
        self.center_points = new_center_points
        return objects_bbs_ids


tracker = Tracker()
class_list = ['car', 'motorcycle', 'bus', 'truck']

st.title("Vehicle Speed Detection")
st.text("Untuk Video LIVE terdapat sedikit problem perhitungan speed karena kualitas resolusi yang dihasilkan live kamera.")
st.text("Untuk mendapatkan hasil yang memuaskan dari model, disarankan menggunakan prerecord video dibawah.")
frame_placeholder = st.empty()

camera_option = st.radio("Select Camera: ([LIVE] require to have a good internet connection to get speed detected)", ["None", "[LIVE] Tol Dalam Kota (Kelapa Gading-Pulo Gebang)", "[LIVE] Jakarta-Bogor-Ciawi", "[LIVE] Jakarta-Cikampek", "[Pre-recorded] Sample Camera 1", "[Pre-recorded] Sample Camera 2", "[Pre-recorded] Sample Camera 3"])

camera_settings = {
    "[LIVE] Tol Dalam Kota (Kelapa Gading-Pulo Gebang)": {
        "url": 'https://camera.jtd.co.id/camera/share/tios/2/78/index.m3u8', #Tol Dalam Kota (Kelapa Gading-Pulo Gebang)
        #realurl: https://bpjt.pu.go.id/cctv/cctv_inframe/?id_ruas=6td&status=online
        # 22+670
        "red_line_y": 300,
        "blue_line_y": 400,
        "lane_divider": 560,
        "distance": 100,
    },
    "[LIVE] Jakarta-Bogor-Ciawi": {
        "url": 'https://jmlive.jasamarga.com/hls/1/099ccdf/index.m3u8',#Jakarta-Bogor-Ciawi
        #realurl: https://bpjt.pu.go.id/cctv/cctv_inframe/?id_ruas=jagorawi&status=online
        #KM 05+500 | B
        "red_line_y": 250,
        "blue_line_y": 400,
        "lane_divider": 520,
        "distance": 100,
    },
    "[LIVE] Jakarta-Cikampek": {
        "url": 'https://jmlive.jasamarga.com/hls/5/31668e25-dc03-4a1c-820c-4a35b1b0dc6c/index.m3u8', #Jakarta-Cikampek
        #realurl: https://bpjt.pu.go.id/cctv/cctv_inframe/?id_ruas=japek&status=online
        #CIKAMPEK KM 03+000 (HALIM - JATIWARINGIN)
        "red_line_y": 250,
        "blue_line_y": 400,
        "lane_divider": 550,
        "distance": 70, 
    },
    "[Pre-recorded] Sample Camera 1": {
        "url": 'sample1.mp4',
        "red_line_y": 250,
        "blue_line_y": 400,
        "lane_divider": 500,
        "distance": 100,
    },
    "[Pre-recorded] Sample Camera 2": {
        "url": 'sample2.mp4',
        "red_line_y": 250,
        "blue_line_y": 400,
        "lane_divider": 500,
        "distance": 140,
    },
    "[Pre-recorded] Sample Camera 3": {
        "url": 'sample3.mp4',
        "red_line_y": 250,
        "blue_line_y": 400,
        "lane_divider": 500,
        "distance": 140,
    }
}

if camera_option != "None":
    selected_camera = camera_settings[camera_option]
    cap = cv2.VideoCapture(selected_camera["url"])
    red_line_y = selected_camera["red_line_y"]
    blue_line_y = selected_camera["blue_line_y"]
    lane_divider = selected_camera["lane_divider"]
    distance = selected_camera["distance"]
else:
    cap = None
    red_line_y = blue_line_y = lane_divider = distance = 0

offset = 15

down = {}
up = {}
counter_down = []
counter_up = []
speeds = {}

while cap and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to fetch video stream. Please check the URL or source.")
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    detections = []

    for index, row in px.iterrows():
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        class_id = int(row[5])
        if class_id in [2, 3, 5, 7]: 
            detected_class = model.names[class_id]  
            detections.append([x1, y1, x2 - x1, y2 - y1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, detected_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    bbox_id = tracker.update(detections)
    for bbox in bbox_id:
        x, y, w, h, obj_id = bbox
        cx, cy = x + w // 2, y + h // 2

        if obj_id not in speeds:
            speeds[obj_id] = 0

        if cx < lane_divider: 
            if blue_line_y - offset < cy < blue_line_y + offset:
                up[obj_id] = time.time()
            if obj_id in up:
                if red_line_y - offset < cy < red_line_y + offset:
                    elapsed_time = time.time() - up[obj_id]
                    if obj_id not in counter_up:
                        counter_up.append(obj_id)
                        if distance is not None:
                            speeds[obj_id] = (distance / elapsed_time) * 3.6
            else:
                speeds[obj_id] = speeds.get(obj_id, 0)

        elif cx >= lane_divider: 
            if red_line_y - offset < cy < red_line_y + offset:
                down[obj_id] = time.time()
            if obj_id in down:
                if blue_line_y - offset < cy < blue_line_y + offset:
                    elapsed_time = time.time() - down[obj_id]
                    if obj_id not in counter_down:
                        counter_down.append(obj_id)
                        if distance is not None:
                            speeds[obj_id] = (distance / elapsed_time) * 3.6
            else:
                speeds[obj_id] = speeds.get(obj_id, 0)

        if speeds[obj_id] == 0:
            speed_text = "Processing Speed"
            text_color = (0, 0, 0) 
        else:
            speed_text = f"Speed: {int(speeds[obj_id])} Km/h"
            text_color = (0, 0, 255) 

        cv2.putText(frame, speed_text, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)


    if red_line_y and blue_line_y and lane_divider:
        cv2.line(frame, (0, red_line_y), (1020, red_line_y), (0, 0, 255), 2) 
        cv2.line(frame, (0, blue_line_y), (1020, blue_line_y), (255, 0, 0), 2) 
        cv2.line(frame, (lane_divider, 0), (lane_divider, 500), (255, 255, 255), 2) 
        cv2.putText(frame, f"Down: {len(counter_down)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Up: {len(counter_up)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_placeholder.image(frame, channels="BGR")
    time.sleep(0.05) 