This project implements a computer vision-based agricultural monitoring system using YOLOv8 for real-time object detection and crop analysis.

The system is designed to assist farmers and agricultural analysts by automating field monitoring, identifying crop conditions, and generating structured insights from visual data.

It transforms raw field images into actionable information such as crop health status, anomaly detection, and productivity estimation.
Key Features
🌾 Real-time crop and object detection using YOLOv8
🧠 Plant health classification (healthy / stressed / diseased)
📊 Field-level analysis and yield estimation insights
📷 Image-based inference pipeline
📍 Scalable design for drone / 360° camera integration
📈 Visualization-ready output for reporting

# 🌱 Agricultural Monitoring System Architecture
from detect import run_detection
from preprocess import preprocess_image
from postprocess import generate_report

def run_pipeline(image_path):
    print("Starting pipeline...")

    # Step 1: Preprocess
    clean_image = preprocess_image(image_path)

    # Step 2: Detection (YOLOv8)
    detections = run_detection(clean_image)

    # Step 3: Post-processing
    report = generate_report(detections)

    print("Pipeline completed")
    return report


if __name__ == "__main__":
    result = run_pipeline("data/input_images/field.jpg")
    print(result)
