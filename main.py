import cv2
from ultralytics import YOLO

model = YOLO("Weights/Best/best500E.pt")

input_video = "vid2.mov"
cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = 'output1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

CONFIDENCE_THRESHOLD = 0.75

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)  

    result = results[0]
    high_conf_boxes = result.boxes[result.boxes.conf > CONFIDENCE_THRESHOLD]

    result.boxes = high_conf_boxes  

    annotated_frame = result.plot()

    out.write(annotated_frame)
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
