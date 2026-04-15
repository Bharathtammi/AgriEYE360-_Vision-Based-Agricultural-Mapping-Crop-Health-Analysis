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

flowchart TD

A[360 Cameras / Drones / Sensors] --> B[Edge Processing]
B --> C[Data Ingestion]
C --> D[Preprocessing Layer]

D --> E[YOLOv8 Object Detection]
D --> F[Crop Health Classification]

E --> G[Feature Extraction]
F --> G

G --> H[Analytics Engine]
H --> I[Yield Prediction]
H --> J[Anomaly Detection]

I --> K[GIS Mapping Layer]
J --> K

K --> L[Path Planning Module]
K --> M[Heatmap Generation]

L --> N[Farmer Dashboard]
M --> N

N --> O[Street View Visualization]
N --> P[Reports & Alerts]
