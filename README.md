# 🌱 Agricultural Monitoring System Architecture

## 📊 Architecture Diagram

```mermaid
flowchart TD
    A[360 Camera / Field Sensors] --> B[Data Acquisition Layer]
    B --> C[Preprocessing Layer]
    C --> D[Detection and Classification]
    D --> E[Plant Health Assessment]
    E --> F[Analytics Engine]
    F --> G[Yield Estimation]
    G --> H[Decision Support System]
    H --> I[Visualization Layer - Street View]
    I --> J[End User - Farmer or Operator]

    %% Supporting Components
    C --> K[Data Cleaning and Synchronization]
    D --> L[Object Detection Module]
    D --> M[Image Classification Module]
    F --> N[Pattern Analysis]
    H --> O[Alerts and Reports]
    I --> P[Field Mapping Interface]
