import cv2
from deepface import DeepFace
import numpy as np

# --- CONFIGURATION ---
FRAME_SKIP = 3 

def run_emotion_debug():
    # 1. Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    
    frame_counter = 0
    current_emotions = {} 

    print("--- EMOTION DEBUGGER RUNNING ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Detect Faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        # Reset offsets for drawing
        y_offset = 60
        x_offset = 20

        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face

            # --- AI LOGIC ---
            if frame_counter % FRAME_SKIP == 0:
                try:
                    margin = 40 
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)
                    
                    face_crop = frame[y1:y2, x1:x2]

                    # Analyze
                    result = DeepFace.analyze(face_crop, 
                                              actions=['emotion'], 
                                              enforce_detection=False, 
                                              detector_backend='opencv', 
                                              silent=True)
                    
                    current_emotions = result[0]['emotion']

                except Exception as e:
                    pass

            # --- VISUALIZATION ---
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # --- STATS BAR ---
            # FIX: Used valid font name here
            cv2.putText(frame, "AI CONFIDENCE:", (x_offset, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Sort emotions by score
            sorted_emotions = sorted(current_emotions.items(), key=lambda item: item[1], reverse=True)

            for emotion, score in sorted_emotions:
                # Green text if score > 30%, Gray if lower
                color = (0, 255, 0) if score > 30 else (200, 200, 200)
                
                text = f"{emotion.upper()}: {score:.1f}%"
                cv2.putText(frame, text, (x_offset, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw Bar Chart
                bar_length = int(score * 2)
                cv2.rectangle(frame, (x_offset + 160, y_offset - 10), 
                              (x_offset + 160 + bar_length, y_offset + 5), color, -1)
                
                y_offset += 30

        cv2.imshow('Emotion Debugger', frame)
        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_emotion_debug()