# ML-CW-NHPT-Project
# AI-Based Surface Defect Detection and Context-Aware Inspection Assistant using YOLOv8 and RAG

## Project Overview

This project was developed as part of the **Machine Learning Applications** coursework. The objective of the system is to automatically detect common surface defects using a fine-tuned YOLOv8 object detection model and provide intelligent inspection assistance through a Retrieval-Augmented Generation (RAG) chatbot.

The developed system identifies three surface conditions:

- Normal
- Crack
- Hole

The application includes a Streamlit-based graphical user interface that allows users to upload inspection images, perform defect detection, visualize prediction results with confidence scores, and interact with an AI assistant for defect-related explanations and inspection guidance.

---

## Key Features

- Automatic surface defect detection using YOLOv8
- Detection of Crack, Hole and Normal surface conditions
- Bounding box visualization with confidence scores
- Real-time image prediction
- AI-powered inspection assistant using RAG
- User-friendly Streamlit web application
- Support for image upload and result visualization

---

## Project Structure

```
ML-CW-NHPT-Project/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
├── .gitignore             # Git ignore configuration
│
├── dataset/
│   └── data.yaml          # Dataset configuration file
│
└── weights/
    └── best.pt            # Trained YOLOv8 model weights
```

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- Ollama
- FAISS
- LangChain

---

## Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/ML-CW-NHPT-Project.git
```

Move into the project directory.

```bash
cd ML-CW-NHPT-Project
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit application.

```bash
streamlit run app.py
```

The application will open automatically in your default web browser.

---

## Model Information

Model: YOLOv8

Framework: Ultralytics YOLO

Training Dataset Classes:

- Normal
- Crack
- Hole

Output:

- Bounding Boxes
- Class Labels
- Confidence Scores

---

## AI Inspection Assistant

The system integrates a Retrieval-Augmented Generation (RAG) assistant powered by Ollama and LangChain.

The assistant helps users by:

- Explaining detected defects
- Providing inspection guidance
- Answering defect-related questions
- Offering contextual information about surface conditions

---

## Repository Contents

This repository contains:

- Source code
- Trained YOLOv8 model
- Dataset configuration file
- Application interface
- Dependency list
- Project documentation

---

## Author

**Lahiru Pramuditha**

Machine Learning Applications Coursework

Department of Data Science

Cobscds252P-027

---

## License

This project was developed solely for academic and educational purposes.
