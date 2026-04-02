import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import tempfile
from PIL import Image
import os

# Load trained model
model = YOLO(r"C:\Users\bodap\OneDrive\Desktop\chandrahas\pothole_detection\train_yolov8\runs\segment\runs\train\yolov8_pothole_seg_custom\weights\best.pt")

st.title("Pothole Detection - Image & Video Dashboard")

uploaded_media = st.file_uploader(
    "Upload Image or Video",
    type=["mp4", "avi", "mov", "jpg", "png", "jpeg"]
)

if uploaded_media:

    file_ext = uploaded_media.name.split(".")[-1].lower()

    results_data = []
    low_count = medium_count = high_count = 0

    # ===================== IMAGE =====================
    if file_ext in ["jpg", "png", "jpeg"]:

        image = Image.open(uploaded_media)
        frame = np.array(image)
        frame_id = uploaded_media.name

        results = model(frame)

        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        detection_id = 0

        for box, conf in zip(boxes, confs):
            detection_id += 1

            x1, y1, x2, y2 = map(int, box)

            width = x2 - x1
            height = y2 - y1
            area = width * height

            # Size classification
            if area < 5000:
                size_label = "Low"
                low_count += 1
            elif area < 15000:
                size_label = "Medium"
                medium_count += 1
            else:
                size_label = "High"
                high_count += 1

            label_text = f"ID:{detection_id} | {size_label} | Conf:{conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label_text,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0,0,0),
                        1)

            results_data.append({
                "Image_ID": frame_id,
                "Detection_ID": detection_id,
                "Size": size_label,
                "Area (pixels)": area,
                "Confidence": round(float(conf), 3)
            })

        st.image(frame)

    # ===================== VIDEO (FINAL FIXED) =====================
    else:

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_media.read())

        cap = cv2.VideoCapture(temp_file.name)
        stframe = st.empty()

        frame_number = 0

        pothole_info = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")

            if results[0].boxes is not None and results[0].boxes.id is not None:

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                for box, conf, track_id in zip(boxes, confs, track_ids):

                    track_id = int(track_id)

                    x1, y1, x2, y2 = map(int, box)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height

                    if area < 5000:
                        size_label = "Low"
                    elif area < 15000:
                        size_label = "Medium"
                    else:
                        size_label = "High"

                    label_text = f"ID:{track_id} | {size_label} | {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, label_text,
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255,0,0),
                                2)

                    # Store only first appearance
                    if track_id not in pothole_info:
                        pothole_info[track_id] = {
                            "Pothole_ID": track_id,
                            "First_Frame": frame_number,
                            "Size": size_label,
                            "Area (pixels)": area,
                            "Confidence": round(float(conf), 3)
                        }

            stframe.image(frame, channels="BGR")

        cap.release()

        results_data = list(pothole_info.values())
    # ===================== TABLE DISPLAY =====================
    df = pd.DataFrame(results_data)

    st.subheader("📊 Detection Table")
    st.dataframe(df)

    st.subheader("📈 Summary")

    if not df.empty:
        low_count = (df["Size"] == "Low").sum()
        medium_count = (df["Size"] == "Medium").sum()
        high_count = (df["Size"] == "High").sum()

        summary_df = pd.DataFrame({
            "Size Category": ["Low", "Medium", "High"],
            "Count": [low_count, medium_count, high_count]
        })

        st.table(summary_df)
    else:
        st.warning("No potholes detected.")