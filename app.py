from ultralytics import YOLO
import cv2

print("Libraries imported successfully...")

# Load a pretrained YOLOv8 model
model = YOLO("D:\\YOLO Test\\yolo11n.pt")  # Use "yolov8s.pt" for better accuracy
print("Model loaded successfully...")

# Choose the source: "image", "video", or "webcam"
source_type = "video"  # Change to "video" or "image" if needed
print(f"Source type selected: {source_type}")

if source_type == "image":
    # Replace with your image file path
    image_path = "D:\\YOLO Test\\548868897_mrcTES7HOdS2hAXW-s-g5HYwuCNO-05gi7EeDLp5oh0.jpg"
           
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    
    # Ensure the image is loaded correctly
    if img is None:
        print("Error: Image not found or could not be read!")
    else:
        # Perform YOLO detection
        results = model(img)

        # Draw bounding boxes on the image
        img = results[0].plot()

        # Resize for better visualization
        img = cv2.resize(img, (800, 600))

        # Display the image
        cv2.imshow("YOLO Image Detection", img)
        cv2.waitKey(0)  # Wait for key press
        cv2.destroyAllWindows()

elif source_type == "video":
    # Replace with your video file path
    video_path = "D:\\YOLO Test\\video.mp4"
    cap = cv2.VideoCapture(video_path)

elif source_type == "webcam":
    # Open webcam (0 for default webcam)
    cap = cv2.VideoCapture(0)

if source_type in ["video", "webcam"]:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Exit if the video file or webcam stream ends

        # Perform YOLO detection on the frame
        results = model(frame)

        # Draw bounding boxes
        frame = results[0].plot()

        # Resize for better viewing
        frame = cv2.resize(frame, (800, 600))

        # Display the frame
        cv2.imshow("YOLO Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 