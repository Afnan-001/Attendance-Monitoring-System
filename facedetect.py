import cv2
import os
import time
from datetime import datetime

def capture_faces_with_toggle(base_output_folder="faces"):
    # Create a new folder dynamically based on the current timestamp
    folder_name = datetime.now().strftime("capture_%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_output_folder, folder_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the DNN face detection model
    model_file = "deploy.prototxt"  # Path to the model definition file
    weights_file = "res10_300x300_ssd_iter_140000.caffemodel"  # Path to the weights file

    if not os.path.exists(model_file) or not os.path.exists(weights_file):
        print("Model or weights file not found. Please check the paths.")
        return

    net = cv2.dnn.readNetFromCaffe(model_file, weights_file)
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Failed to open camera.")
        return

    cv2.namedWindow("Capture Faces")
    capturing = False  # Toggle state
    captured_count = 0
    max_photos = 500  # Maximum number of photos to capture
    last_capture_time = 0  # Track the last capture time

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        current_time = time.time()

        if capturing and current_time - last_capture_time >= 2 and captured_count < max_photos:  # 2-second interval
            # Prepare frame for DNN model
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.6:  # Confidence threshold
                    # Get coordinates of the detected face
                    h, w = frame.shape[:2]
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    (x1, y1, x2, y2) = box.astype("int")
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    # Crop and save the face
                    face = frame[y1:y2, x1:x2]
                    if face.size > 0:
                        face = cv2.resize(face, (128, 128))
                        face_path = os.path.join(output_folder, f"face_{captured_count}_{int(time.time())}.jpg")
                        cv2.imwrite(face_path, face)
                        print(f"Saved: {face_path}")
                        captured_count += 1

                    # Draw rectangle (optional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Update the last capture time
                    last_capture_time = current_time

                    # Stop capturing if the maximum number of photos is reached
                    if captured_count >= max_photos:
                        print("Captured 500 photos. Exiting...")
                        capturing = False
                        break

        # Display instructions and toggle state
        toggle_status = "ON" if capturing else "OFF"
        cv2.putText(frame, f"Face Capture: {toggle_status} (Press SPACE to toggle, ESC to exit)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if capturing else (0, 0, 255), 2)

        cv2.imshow("Capture Faces", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to exit
            print("Exiting...")
            break
        elif key == 32:  # SPACE to toggle
            capturing = not capturing
            print(f"Face capture toggled {'ON' if capturing else 'OFF'}.")

    cam.release()
    cv2.destroyAllWindows()

# Call the function
capture_faces_with_toggle()
