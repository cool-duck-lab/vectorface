import face_recognition
import cv2
import numpy as np

# =========================
# 1. טעינת תמונת הדוגמה
# =========================
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# =========================
# 2. פתיחת מצלמה
# =========================
video_capture = cv2.VideoCapture(0)

print("המצלמה הופעלה. לחץ q ליציאה")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # המרה מ־BGR ל־RGB
    rgb_frame = frame[:, :, ::-1]

    # =========================
    # 3. זיהוי פנים בפריים
    # =========================
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        # =========================
        # 4. השוואת וקטורים
        # =========================
        distance = face_recognition.face_distance(
            [known_encoding],
            face_encoding
        )[0]

        is_match = distance < 0.5  # סף (אפשר לכוונן)

        name = "MATCH" if is_match else "UNKNOWN"
        color = (0, 255, 0) if is_match else (0, 0, 255)

        # ציור מלבן
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(
            frame,
            f"{name} ({distance:.2f})",
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
