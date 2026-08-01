# ML-CW-NHPT-Project
# AI-Based Surface Defect Detection and Context-Aware Inspection Assistant using YOLOv8 and RAG

## Project Overview

This project was developed as part of the Machine Learning Applications coursework.

The system detects three surface conditions:

- Normal
- Crack
- Hole

using a fine-tuned YOLOv8 object detection model.

A Retrieval-Augmented Generation (RAG) assistant powered by Ollama is integrated to provide inspection guidance and explanations based on the detected defects.

---

## Features

- Surface defect detection using YOLOv8
- Detection of Crack, Hole and Normal classes
- Bounding box visualization
- Confidence score display
- AI Inspection Assistant (RAG)
- Streamlit Web Interface

---

## Project Structure

ML-CW-NHPT-Project/

 app.py >
 >requirements.txt
 >README.md
 >dataset/
      data.yaml
 >weights/
      best.pt

## Technologies

Python
Streamlit
YOLOv8
Ultralytics
OpenCV
Ollama
FAISS
LangChain

## Author
CTL Pramuditha
