#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

print(sys.version)
print(sys.executable)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


import numpy as np

print(np.__version__)
print(np.__file__)


# In[3]:


import cv2

print(cv2.__version__)


# In[4]:


from ultralytics import YOLO

print("Ultralytics Loaded Successfully!")


# In[5]:


import torch

print(torch.__version__)
print(torch.cuda.is_available())


# In[7]:


import os

print(os.getcwd())


# In[8]:


import os

print(os.listdir())


# In[9]:


import os

for folder in ["Desktop", "OneDrive"]:
    path = os.path.join(os.getcwd(), folder)
    if os.path.exists(path):
        print(f"\nContents of {path}:")
        print(os.listdir(path)[:30])   # first 30 items


# In[19]:


from ultralytics import YOLO

model = YOLO("NHPT_Project_v2/weights/best.pt")

print("Model Loaded Successfully!")


# In[20]:


results = model.predict(
    source="Crack_Hole_Normal_Dataset/images/test/00016.jpg",
    save=True,
    conf=0.25
)

print("Prediction Completed!")


# In[21]:


import glob

predicted_images = glob.glob("runs/detect/predict/*.jpg")
print(predicted_images)


# In[22]:


from PIL import Image
import matplotlib.pyplot as plt

img = Image.open(predicted_images[0])

plt.figure(figsize=(10,8))
plt.imshow(img)
plt.axis("off")
plt.show()


# In[23]:


from IPython.display import display
from PIL import Image

img = Image.open(predicted_images[0])
display(img)


# In[24]:


metrics = model.val(data="Crack_Hole_Normal_Dataset/data.yaml")


# In[25]:


import os

for cls in ["train", "test"]:
    print("="*40)
    print(cls)

    labels = os.listdir(f"Crack_Hole_Normal_Dataset/labels/{cls}")

    normal = 0
    crack = 0
    hole = 0

    for file in labels:
        with open(f"Crack_Hole_Normal_Dataset/labels/{cls}/{file}") as f:
            for line in f:
                c = int(line.split()[0])

                if c == 0:
                    normal += 1
                elif c == 1:
                    crack += 1
                elif c == 2:
                    hole += 1

    print("Normal =", normal)
    print("Crack  =", crack)
    print("Hole   =", hole)


# In[26]:


import os

print(os.path.exists("Crack_Hole_Normal_Dataset/images/val"))


# In[27]:


import os

if os.path.exists("Crack_Hole_Normal_Dataset/images/val"):
    print("Validation Images:", len(os.listdir("Crack_Hole_Normal_Dataset/images/val")))

if os.path.exists("Crack_Hole_Normal_Dataset/labels/val"):
    print("Validation Labels:", len(os.listdir("Crack_Hole_Normal_Dataset/labels/val")))


# In[28]:


import glob

label_files = glob.glob("Crack_Hole_Normal_Dataset/labels/train/*.txt")

for f in label_files[:10]:
    print("FILE:", f)

    with open(f) as file:
        print(file.readline())


# In[29]:


import glob
from collections import Counter

counter = Counter()

label_files = glob.glob("Crack_Hole_Normal_Dataset/labels/train/*.txt")

for file in label_files:
    with open(file, "r") as f:
        for line in f:
            if line.strip():
                cls = int(line.split()[0])
                counter[cls] += 1

print(counter)


# In[30]:


import os

os.makedirs("Crack_Hole_Normal_Dataset/images/val", exist_ok=True)
os.makedirs("Crack_Hole_Normal_Dataset/labels/val", exist_ok=True)

print("Folders Created!")


# In[32]:


import os
from collections import Counter

extensions = Counter()

for img in os.listdir("Crack_Hole_Normal_Dataset/images/train"):
    ext = os.path.splitext(img)[1].lower()
    extensions[ext] += 1

print(extensions)


# In[33]:


import os
import shutil
import random

random.seed(42)

train_img = "Crack_Hole_Normal_Dataset/images/train"
train_lbl = "Crack_Hole_Normal_Dataset/labels/train"

val_img = "Crack_Hole_Normal_Dataset/images/val"
val_lbl = "Crack_Hole_Normal_Dataset/labels/val"

os.makedirs(val_img, exist_ok=True)
os.makedirs(val_lbl, exist_ok=True)

images = os.listdir(train_img)
random.shuffle(images)

num_val = int(len(images) * 0.20)

count = 0

for img in images[:num_val]:

    name = os.path.splitext(img)[0]
    label = name + ".txt"

    src_img = os.path.join(train_img, img)
    dst_img = os.path.join(val_img, img)

    src_lbl = os.path.join(train_lbl, label)
    dst_lbl = os.path.join(val_lbl, label)

    shutil.copy2(src_img, dst_img)

    if os.path.exists(src_lbl):
        shutil.copy2(src_lbl, dst_lbl)
        count += 1

print("Validation Images :", len(os.listdir(val_img)))
print("Validation Labels :", len(os.listdir(val_lbl)))
print("Copied Successfully!")


# In[35]:


import os

print(os.path.exists("Crack_Hole_Normal_Dataset/images/train"))
print(os.path.exists("Crack_Hole_Normal_Dataset/images/val"))
print(os.path.exists("Crack_Hole_Normal_Dataset/images/test"))


# In[36]:


from ultralytics import YOLO

model = YOLO("C:/Users/HP/Downloads/NHPT_Project_v2/weights/best.pt")

model.train(
    data="Crack_Hole_Normal_Dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    project="NHPT_Project_v2",
    name="Fine_Tuned",
    workers=0
)


# In[37]:


from ultralytics import YOLO

model = YOLO(r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt")

model.predict(
    source=r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset\images\test",
    save=True,
    conf=0.25
)


# In[38]:


from ultralytics import YOLO

# Load best model
model = YOLO(r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt")

# Evaluate model
metrics = model.val(
    data=r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset\data.yaml"
)

print(metrics)


# In[39]:


from ultralytics import YOLO

model = YOLO(r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt")

results = model.predict(
    source=r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset\images\test",
    save=True,
    conf=0.25,
    show=False
)


# In[43]:


import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[44]:


import os

print(os.path.exists(os.path.join(results_path, "results.png")))


# In[ ]:





# In[41]:


results_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2"


# In[46]:


img = mpimg.imread(os.path.join(results_path, "results.png"))

plt.figure(figsize=(16,8))
plt.imshow(img)
plt.axis("off")
plt.title("Training Results")
plt.show()


# In[47]:


img = mpimg.imread(os.path.join(results_path, "BoxP_curve.png"))

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.title("Precision Curve")
plt.show()


# In[48]:


img = mpimg.imread(os.path.join(results_path, "BoxR_curve.png"))

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.title("Recall Curve")
plt.show()


# In[49]:


img = mpimg.imread(os.path.join(results_path, "BoxF1_curve.png"))

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.title("F1 Score Curve")
plt.show()


# In[50]:


img = mpimg.imread(os.path.join(results_path, "BoxPR_curve.png"))

plt.figure(figsize=(8,6))
plt.imshow(img)
plt.axis("off")
plt.title("Precision-Recall Curve")
plt.show()


# In[51]:


img = mpimg.imread(os.path.join(results_path, "confusion_matrix.png"))

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.axis("off")
plt.title("Confusion Matrix")
plt.show()


# In[52]:


img = mpimg.imread(os.path.join(results_path, "confusion_matrix_normalized.png"))

plt.figure(figsize=(8,8))
plt.imshow(img)
plt.axis("off")
plt.title("Normalized Confusion Matrix")
plt.show()


# In[53]:


img = mpimg.imread(os.path.join(results_path, "labels.jpg"))

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.title("Dataset Labels")
plt.show()


# In[1]:


import os

project_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"

os.chdir(project_path)

print("Current Project Directory:")
print(os.getcwd())


# In[2]:


import os

model_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"

print("Model exists:", os.path.exists(model_path))


# In[3]:


import importlib.util

libraries = {
    "ultralytics": "YOLO / Computer Vision",
    "faiss": "FAISS Vector Search",
    "sentence_transformers": "Sentence Embeddings",
    "langchain": "RAG Framework",
    "streamlit": "Web Application",
    "ollama": "Local LLM Connection"
}

print("=== NHPT PROJECT ENVIRONMENT CHECK ===\n")

for library, purpose in libraries.items():
    if importlib.util.find_spec(library):
        print(f"✅ {library:<25} INSTALLED  | {purpose}")
    else:
        print(f"❌ {library:<25} NOT INSTALLED | {purpose}")


# In[4]:


import subprocess

print("=== AVAILABLE OLLAMA MODELS ===\n")

result = subprocess.run(
    ["ollama", "list"],
    capture_output=True,
    text=True
)

print(result.stdout)

if result.stderr:
    print("Error:", result.stderr)


# In[5]:


from ultralytics import YOLO
import os

# Path to the trained YOLO model
model_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"

# Check whether the trained model exists
print("Model exists:", os.path.exists(model_path))

# Load the trained model
model = YOLO(model_path)

print("Trained YOLO model loaded successfully!")
print("Detected Classes:", model.names)


# In[6]:


# Stage 3.2 - Test the trained YOLO model on a test image

import matplotlib.pyplot as plt
import cv2
import os

# Test image path
test_image_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset\images\test\97514717.png"

# Check whether test image exists
print("Test image exists:", os.path.exists(test_image_path))

# Run prediction using the already loaded trained model
results = model.predict(
    source=test_image_path,
    conf=0.25,
    save=False
)

# Get the prediction result
result = results[0]

# Plot prediction with bounding boxes
annotated_image = result.plot()

# Convert BGR to RGB for Matplotlib
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display prediction result
plt.figure(figsize=(10, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.title("YOLO Model Prediction Result")
plt.show()


# In[7]:


# Stage 3.3 - Extract detected classes and confidence scores

print("=== YOLO DETECTION RESULTS ===\n")

detections = []

# Extract bounding box results
for box in result.boxes:
    
    # Get class ID
    class_id = int(box.cls[0])
    
    # Get class name
    class_name = model.names[class_id]
    
    # Get confidence score
    confidence = float(box.conf[0])
    
    # Store detection
    detections.append({
        "class": class_name,
        "confidence": confidence
    })
    
    # Display result
    print(f"Detected Class: {class_name}")
    print(f"Confidence Score: {confidence:.2f}")
    print("-----------------------------")

print("\nTotal Detections:", len(detections))


# In[8]:


# Stage 4.1 - Create Domain Knowledge Base for RAG

knowledge_documents = [
    """
    Crack Detection:
    A crack is a visible fracture or narrow separation that appears on the surface
    of a material or structure. Cracks may indicate structural deterioration or
    damage. When a crack is detected, the affected area should be inspected to
    determine its severity, length, width, and possible progression.
    """,

    """
    Hole Detection:
    A hole is an opening, cavity, or missing section detected on the surface of
    a material or structure. Holes may occur due to physical damage, deterioration,
    impact, or material loss. A detected hole should be inspected to determine
    its size, depth, and potential impact on the surrounding structure.
    """,

    """
    Normal Surface:
    A normal surface is an area where no significant crack or hole has been
    detected by the computer vision model. The surface appears to have no visible
    structural defect belonging to the trained defect classes. Routine inspection
    and monitoring may still be performed as part of regular maintenance.
    """,

    """
    Inspection Recommendation:
    Computer vision detections should be treated as automated inspection support.
    High-confidence detections can be prioritized for further inspection.
    Lower-confidence detections may require manual visual verification.
    Final decisions regarding structural condition or repair should be made
    after appropriate professional inspection.
    """
]

print("=== RAG KNOWLEDGE BASE ===")
print("Number of knowledge documents:", len(knowledge_documents))

for i, document in enumerate(knowledge_documents, start=1):
    print(f"\nDocument {i}:")
    print(document.strip())


# In[9]:


# Stage 4.2 - Create Embeddings and FAISS Vector Index

import ollama
import faiss
import numpy as np

print("=== CREATING RAG VECTOR DATABASE ===\n")

embeddings = []

# Generate an embedding for each knowledge document
for i, document in enumerate(knowledge_documents):

    response = ollama.embeddings(
        model="nomic-embed-text:latest",
        prompt=document
    )

    embedding = response["embedding"]
    embeddings.append(embedding)

    print(f"Document {i + 1} embedded successfully.")

# Convert embeddings to NumPy array
embedding_matrix = np.array(embeddings).astype("float32")

print("\nEmbedding Matrix Shape:", embedding_matrix.shape)

# Get embedding dimension
dimension = embedding_matrix.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)

# Add document embeddings to FAISS
index.add(embedding_matrix)

print("FAISS Index Created Successfully!")
print("Total Documents in FAISS Index:", index.ntotal)


# In[11]:


# Stage 4.3 - Test Semantic Retrieval from FAISS

def retrieve_relevant_knowledge(query, top_k=2):
    """
    Retrieve the most relevant knowledge documents
    from the FAISS vector index.
    """

    # Convert the query into an embedding
    response = ollama.embeddings(
        model="nomic-embed-text:latest",
        prompt=query
    )

    query_embedding = np.array(
        [response["embedding"]]
    ).astype("float32")

    # Search the FAISS index
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    # Retrieve matching documents
    retrieved_documents = []

    for rank, doc_index in enumerate(indices[0]):
        retrieved_documents.append(
            knowledge_documents[doc_index]
        )

        print(f"=== Retrieved Document {rank + 1} ===")
        print(f"FAISS Distance: {distances[0][rank]:.4f}")
        print(knowledge_documents[doc_index].strip())
        print()

    return retrieved_documents


# In[12]:


# Test retrieval using a YOLO detection result

test_query = """
The computer vision model detected a crack
with a confidence score of 0.82.
What information is relevant to this detection?
"""

print("=== SEMANTIC RETRIEVAL TEST ===")
print("Query:")
print(test_query)

print("\nRetrieving relevant knowledge...\n")

retrieved_context = retrieve_relevant_knowledge(
    test_query,
    top_k=2
)


# In[13]:


# Stage 4.4 - Generate a RAG response using the local Ollama LLM

import ollama

print("=== RAG RESPONSE GENERATION ===\n")

# Combine the documents retrieved from FAISS
context = "\n\n".join(retrieved_context)

# Detection information from the YOLO model
detection_summary = """
The computer vision model detected:
- 1 crack with confidence score 0.82
- 4 holes with confidence scores 0.64, 0.55, 0.40, and 0.35
"""

# Create the prompt using YOLO output and retrieved knowledge
prompt = f"""
You are an AI assistant supporting automated surface defect inspection.

Computer Vision Detection Results:
{detection_summary}

Retrieved Domain Knowledge:
{context}

Based only on the detection results and retrieved domain knowledge above,
provide:
1. A short summary of the detected defects.
2. An explanation of what the detected defects may indicate.
3. A suitable inspection recommendation.

Do not claim that the structure is safe or unsafe.
Keep the response concise and clear.
"""

print("Generating response using local Ollama LLM...\n")

# Generate response using the locally installed LLM
response = ollama.chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# Extract generated response
rag_response = response["message"]["content"]

print("=== GENERATED RAG RESPONSE ===\n")
print(rag_response)


# In[14]:


# Stage 5.1 - Full Integrated CV + RAG + LLM Pipeline

import cv2
import numpy as np
import ollama

def analyze_surface_image(image_path):
    
    print("=== STARTING INTEGRATED ANALYSIS ===\n")

    # -------------------------------------------------
    # STEP 1: Run YOLO Computer Vision Model
    # -------------------------------------------------
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]

    # -------------------------------------------------
    # STEP 2: Extract Detection Results
    # -------------------------------------------------
    detection_list = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        detection_list.append({
            "class": class_name,
            "confidence": confidence
        })

    # Create detection summary
    if detection_list:

        detection_summary = "\n".join(
            [
                f"- {item['class']} detected with confidence {item['confidence']:.2f}"
                for item in detection_list
            ]
        )

    else:
        detection_summary = "No defects were detected by the computer vision model."

    print("=== COMPUTER VISION RESULTS ===")
    print(detection_summary)

    # -------------------------------------------------
    # STEP 3: Retrieve Relevant Knowledge from FAISS
    # -------------------------------------------------
    retrieval_query = (
        "Computer vision surface inspection results:\n"
        + detection_summary
    )

    embedding_response = ollama.embeddings(
        model="nomic-embed-text:latest",
        prompt=retrieval_query
    )

    query_embedding = np.array(
        [embedding_response["embedding"]]
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        2
    )

    retrieved_documents = [
        knowledge_documents[i]
        for i in indices[0]
    ]

    context = "\n\n".join(retrieved_documents)

    print("\n=== RELEVANT KNOWLEDGE RETRIEVED ===")
    print(context)

    # -------------------------------------------------
    # STEP 4: Generate RAG Response using Ollama
    # -------------------------------------------------
    prompt = f"""
You are an AI assistant supporting automated surface defect inspection.

Computer Vision Detection Results:
{detection_summary}

Retrieved Domain Knowledge:
{context}

Based only on the information provided above:

1. Summarize the detected defects.
2. Explain what the detections may indicate.
3. Provide an appropriate inspection recommendation.

Do not claim that the structure is safe or unsafe.
Keep the response concise and clear.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    ai_response = response["message"]["content"]

    # -------------------------------------------------
    # STEP 5: Prepare Annotated Prediction Image
    # -------------------------------------------------
    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    print("\n=== AI GENERATED INSPECTION REPORT ===")
    print(ai_response)

    # Return outputs for the application
    return annotated_image, ai_response


# In[15]:


# Stage 5.1 - Test the Full Integrated Pipeline

annotated_output, inspection_report = analyze_surface_image(
    test_image_path
)


# In[17]:


from collections import Counter
import matplotlib.pyplot as plt
import cv2
import numpy as np
import ollama


def analyze_surface_image_v2(image_path):

    print("=== STARTING IMPROVED INTEGRATED ANALYSIS ===\n")

    # -------------------------------------------------
    # STEP 1: YOLO Prediction
    # -------------------------------------------------
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False,
        verbose=False
    )

    result = results[0]

    # -------------------------------------------------
    # STEP 2: Extract Exact Detection Results
    # -------------------------------------------------
    detection_list = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        detection_list.append({
            "class": class_name,
            "confidence": confidence
        })

    # Count each detected class exactly
    class_counts = Counter(
        item["class"] for item in detection_list
    )

    # -------------------------------------------------
    # STEP 3: Create Exact Detection Summary
    # -------------------------------------------------
    if detection_list:

        summary_lines = []

        for class_name, count in class_counts.items():

            confidences = [
                item["confidence"]
                for item in detection_list
                if item["class"] == class_name
            ]

            confidence_text = ", ".join(
                f"{conf:.2f}" for conf in confidences
            )

            summary_lines.append(
                f"- {class_name}: {count} detection(s) "
                f"(confidence scores: {confidence_text})"
            )

        detection_summary = "\n".join(summary_lines)

    else:
        detection_summary = "No crack or hole defects were detected."

    print("=== EXACT COMPUTER VISION RESULTS ===")
    print(detection_summary)

    # -------------------------------------------------
    # STEP 4: Retrieve Knowledge for Detected Classes
    # -------------------------------------------------
    retrieved_documents = []

    if detection_list:

        # Retrieve relevant knowledge separately
        # for each unique detected class
        for class_name in class_counts.keys():

            query = (
                f"Information and inspection guidance "
                f"for detected {class_name} defects."
            )

            embedding_response = ollama.embeddings(
                model="nomic-embed-text:latest",
                prompt=query
            )

            query_embedding = np.array(
                [embedding_response["embedding"]]
            ).astype("float32")

            distances, indices = index.search(
                query_embedding,
                1
            )

            document = knowledge_documents[
                indices[0][0]
            ]

            if document not in retrieved_documents:
                retrieved_documents.append(document)

        # Always include inspection guidance
        inspection_document = knowledge_documents[3]

        if inspection_document not in retrieved_documents:
            retrieved_documents.append(inspection_document)

    else:

        # Retrieve normal-surface knowledge
        retrieved_documents.append(
            knowledge_documents[2]
        )

    context = "\n\n".join(retrieved_documents)

    print("\n=== RELEVANT KNOWLEDGE RETRIEVED ===")

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):
        print(f"\nSource {i}:")
        print(document.strip())

    # -------------------------------------------------
    # STEP 5: Generate Grounded LLM Response
    # -------------------------------------------------
    prompt = f"""
You are an AI assistant supporting automated surface defect inspection.

EXACT COMPUTER VISION RESULTS:
{detection_summary}

RETRIEVED DOMAIN KNOWLEDGE:
{context}

IMPORTANT INSTRUCTIONS:

- The computer vision results above are the authoritative detection results.
- Do not change, estimate, or invent the number of detections.
- Do not change any confidence score.
- Report the detected classes and counts exactly as provided.
- Base your explanation only on the retrieved domain knowledge.
- Do not claim that the structure is safe or unsafe.
- Do not make a structural diagnosis.

Provide:

1. Detection Summary
2. Contextual Explanation
3. Inspection Recommendation

Keep the response clear and concise.
"""

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    ai_response = response["message"]["content"]

    print("\n=== AI GENERATED INSPECTION REPORT ===")
    print(ai_response)

    # -------------------------------------------------
    # STEP 6: Prepare Annotated Image
    # -------------------------------------------------
    annotated_image = result.plot()

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    return (
        annotated_image,
        detection_summary,
        ai_response
    )


# In[18]:


annotated_output, exact_detection_summary, inspection_report = \
    analyze_surface_image_v2(test_image_path)


# In[19]:


# Display the annotated YOLO prediction

plt.figure(figsize=(12, 8))
plt.imshow(annotated_output)
plt.axis("off")
plt.title("Integrated YOLO Defect Detection Result")
plt.show()

print("\n=== EXACT DETECTION SUMMARY ===")
print(exact_detection_summary)

print("\n=== AI INSPECTION REPORT ===")
print(inspection_report)


# In[20]:


app_code = """
import streamlit as st

st.set_page_config(
    page_title="AI Surface Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)

st.title("AI Surface Defect Inspection System")

st.write(
    "Upload a surface image to detect cracks and holes "
    "using the trained YOLO computer vision model."
)

st.info(
    "The system integrates Computer Vision, "
    "RAG-based knowledge retrieval, and a local LLM "
    "to provide contextual inspection support."
)

uploaded_file = st.file_uploader(
    "Upload an image for inspection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.success("Image uploaded successfully!")

"""

with open("app.py", "w", encoding="utf-8") as file:
    file.write(app_code)

print("Streamlit app.py created successfully!")


# In[21]:


# Read the current app.py file
with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

# Add YOLO import
app_code = app_code.replace(
    "import streamlit as st",
    """import streamlit as st
from ultralytics import YOLO
import os"""
)

# Add trained model loading code before the title
model_loading_code = r'''
# Path to the trained YOLO model
model_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"

@st.cache_resource
def load_yolo_model():
    return YOLO(model_path)

# Load trained model
model = load_yolo_model()

'''

app_code = app_code.replace(
    'st.title("AI Surface Defect Inspection System")',
    model_loading_code +
    '\nst.title("AI Surface Defect Inspection System")'
)

# Save updated app.py
with open("app.py", "w", encoding="utf-8") as file:
    file.write(app_code)

print("YOLO model integration added to app.py successfully!")
print("Model path exists:", os.path.exists(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"
))


# In[22]:


# Read current app.py
with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

old_code = '''if uploaded_file is not None:
    st.success("Image uploaded successfully!")
'''

new_code = '''if uploaded_file is not None:

    st.success("Image uploaded successfully!")

    # Save uploaded image temporarily
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_image_path = temp_file.name

    # Display original image
    st.subheader("Uploaded Image")

    st.image(
        uploaded_file,
        caption="Original Uploaded Image"
    )

    # Run YOLO prediction
    with st.spinner("Analyzing image using the trained YOLO model..."):

        results = model.predict(
            source=temp_image_path,
            conf=0.25,
            save=False,
            verbose=False
        )

    result = results[0]

    # Create annotated prediction image
    annotated_image = result.plot()

    # Convert BGR to RGB
    annotated_image = annotated_image[:, :, ::-1]

    # Display YOLO detection result
    st.subheader("YOLO Defect Detection Result")

    st.image(
        annotated_image,
        caption="Detected Surface Defects"
    )

    st.success("YOLO analysis completed successfully!")

'''

app_code = app_code.replace(
    old_code,
    new_code
)

# Save updated app.py
with open("app.py", "w", encoding="utf-8") as file:
    file.write(app_code)

print("Image upload and YOLO prediction added successfully!")


# In[23]:


# Stage 6.4 - Add Exact Detection Summary to Streamlit App

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

# Add Counter import
app_code = app_code.replace(
    "import os",
    """import os
from collections import Counter"""
)

# Find the YOLO completion message and add detection summary before it
old_code = '''    st.success("YOLO analysis completed successfully!")'''

new_code = '''    # ---------------------------------------------
    # Extract Exact Detection Results
    # ---------------------------------------------
    detection_list = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        detection_list.append({
            "class": class_name,
            "confidence": confidence
        })

    # Count detections by class
    class_counts = Counter(
        item["class"] for item in detection_list
    )

    # Create exact detection summary
    summary_lines = []

    if detection_list:

        for class_name, count in class_counts.items():

            confidences = [
                item["confidence"]
                for item in detection_list
                if item["class"] == class_name
            ]

            confidence_text = ", ".join(
                f"{conf:.2f}" for conf in confidences
            )

            summary_lines.append(
                f"{class_name}: {count} detection(s) "
                f"(confidence scores: {confidence_text})"
            )

        detection_summary = "\\n".join(summary_lines)

    else:
        detection_summary = "No crack or hole defects were detected."

    # Display exact detection summary
    st.subheader("Exact Detection Summary")

    if detection_list:

        for line in summary_lines:
            st.write("• " + line)

        st.write(
            f"**Total detections:** {len(detection_list)}"
        )

    else:
        st.success(
            "No crack or hole defects were detected."
        )

    st.success("YOLO analysis completed successfully!")'''

app_code = app_code.replace(
    old_code,
    new_code
)

# Save updated app.py
with open("app.py", "w", encoding="utf-8") as file:
    file.write(app_code)

print("Exact detection summary added to app.py successfully!")


# In[24]:


# Stage 6.5 - Add FAISS + RAG Knowledge Retrieval to Streamlit App

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# -------------------------------------------------
# 1. Add required imports
# -------------------------------------------------

app_code = app_code.replace(
    "from collections import Counter",
    """from collections import Counter
import ollama
import faiss
import numpy as np"""
)


# -------------------------------------------------
# 2. Add Knowledge Base + FAISS initialization
#    before the Streamlit title
# -------------------------------------------------

rag_setup_code = r'''
# =================================================
# RAG KNOWLEDGE BASE
# =================================================

knowledge_documents = [

    """
    Crack Detection:
    A crack is a visible fracture or narrow separation that appears on the surface
    of a material or structure. Cracks may indicate structural deterioration or
    damage. When a crack is detected, the affected area should be inspected to
    determine its severity, length, width, and possible progression.
    """,

    """
    Hole Detection:
    A hole is an opening, cavity, or missing section detected on the surface of
    a material or structure. Holes may occur due to physical damage, deterioration,
    impact, or material loss. A detected hole should be inspected to determine
    its size, depth, and potential impact on the surrounding structure.
    """,

    """
    Normal Surface:
    A normal surface is an area where no significant crack or hole has been
    detected by the computer vision model. The surface appears to have no visible
    structural defect belonging to the trained defect classes. Routine inspection
    and monitoring may still be performed as part of regular maintenance.
    """,

    """
    Inspection Recommendation:
    Computer vision detections should be treated as automated inspection support.
    High-confidence detections can be prioritized for further inspection.
    Lower-confidence detections may require manual visual verification.
    Final decisions regarding structural condition or repair should be made
    after appropriate professional inspection.
    """
]


# =================================================
# CREATE FAISS VECTOR INDEX
# =================================================

@st.cache_resource
def create_faiss_index():

    embeddings = []

    for document in knowledge_documents:

        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=document
        )

        embeddings.append(
            response["embedding"]
        )

    embedding_matrix = np.array(
        embeddings
    ).astype("float32")

    dimension = embedding_matrix.shape[1]

    faiss_index = faiss.IndexFlatL2(
        dimension
    )

    faiss_index.add(
        embedding_matrix
    )

    return faiss_index


# Initialize FAISS index
faiss_index = create_faiss_index()

'''


# Insert RAG setup before app title
app_code = app_code.replace(
    'st.title("AI Surface Defect Inspection System")',
    rag_setup_code +
    '\nst.title("AI Surface Defect Inspection System")'
)


# -------------------------------------------------
# 3. Add RAG retrieval after detection summary
# -------------------------------------------------

old_code = '''    st.success("YOLO analysis completed successfully!")'''


new_code = '''    st.success("YOLO analysis completed successfully!")

    # =================================================
    # RAG KNOWLEDGE RETRIEVAL
    # =================================================

    retrieved_documents = []

    if detection_list:

        # Retrieve relevant knowledge separately
        # for every detected class
        for class_name in class_counts.keys():

            retrieval_query = (
                f"Information and inspection guidance "
                f"for detected {class_name} defects."
            )

            embedding_response = ollama.embeddings(
                model="nomic-embed-text:latest",
                prompt=retrieval_query
            )

            query_embedding = np.array(
                [embedding_response["embedding"]]
            ).astype("float32")

            distances, indices = faiss_index.search(
                query_embedding,
                1
            )

            retrieved_document = knowledge_documents[
                indices[0][0]
            ]

            if retrieved_document not in retrieved_documents:

                retrieved_documents.append(
                    retrieved_document
                )

        # Add general inspection recommendation
        inspection_document = knowledge_documents[3]

        if inspection_document not in retrieved_documents:

            retrieved_documents.append(
                inspection_document
            )

    else:

        # If no defects are detected,
        # use normal-surface knowledge
        retrieved_documents.append(
            knowledge_documents[2]
        )


    # Combine retrieved documents
    retrieved_context = "\\n\\n".join(
        retrieved_documents
    )


    # Display retrieved RAG knowledge
    st.subheader(
        "Retrieved Knowledge (RAG)"
    )

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        with st.expander(
            f"Retrieved Knowledge Source {i}"
        ):

            st.write(
                document.strip()
            )
'''


app_code = app_code.replace(
    old_code,
    new_code
)


# -------------------------------------------------
# 4. Save updated app.py
# -------------------------------------------------

with open(
    "app.py",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        app_code
    )


print(
    "FAISS and RAG knowledge retrieval "
    "added to app.py successfully!"
)


# In[25]:


# Stage 6.6 - Add Ollama Local LLM to Streamlit App

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# Find the end of the RAG display section
old_code = '''            st.write(
                document.strip()
            )
'''


# Add LLM response generation after RAG retrieval
new_code = '''            st.write(
                document.strip()
            )


    # =================================================
    # OLLAMA LOCAL LLM - RAG RESPONSE GENERATION
    # =================================================

    st.subheader("AI Inspection Report")

    with st.spinner(
        "Generating contextual inspection report..."
    ):

        rag_prompt = f"""
You are an AI assistant supporting automated surface defect inspection.

EXACT COMPUTER VISION RESULTS:
{detection_summary}

RETRIEVED DOMAIN KNOWLEDGE:
{retrieved_context}

IMPORTANT INSTRUCTIONS:

- Treat the computer vision results as authoritative.
- Do not change or invent the number of detections.
- Do not change any confidence scores.
- Report detected classes and counts exactly as provided.
- Base the explanation only on the retrieved domain knowledge.
- Do not claim that the structure is safe or unsafe.
- Do not make a definitive structural diagnosis.

Provide the response using these three sections:

1. Detection Summary
2. Contextual Explanation
3. Inspection Recommendation

Keep the response clear and concise.
"""

        llm_response = ollama.chat(
            model="qwen2.5:3b",
            messages=[
                {
                    "role": "user",
                    "content": rag_prompt
                }
            ]
        )

        ai_inspection_report = (
            llm_response["message"]["content"]
        )


    # Display AI generated report
    st.markdown(
        ai_inspection_report
    )


    st.success(
        "Integrated CV + RAG + LLM analysis completed!"
    )
'''


app_code = app_code.replace(
    old_code,
    new_code
)


# Save updated app.py
with open(
    "app.py",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        app_code
    )


print(
    "Ollama Local LLM integration "
    "added to app.py successfully!"
)


# In[26]:


# Stage 6.7 - Check app.py for Python syntax errors

import py_compile

try:
    py_compile.compile(
        "app.py",
        doraise=True
    )

    print("SUCCESS: app.py syntax check passed!")
    print("The Streamlit application is ready for the next testing stage.")

except py_compile.PyCompileError as e:

    print("ERROR: Syntax error found in app.py")
    print(e)


# In[27]:


import sys
print(sys.executable)


# In[61]:


from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate


# In[28]:


import importlib.util

packages = {
    "langchain": "LangChain Framework",
    "langchain_community": "LangChain Community Integrations",
    "faiss": "FAISS Vector Search",
    "streamlit": "Streamlit Web Application"
}

print("=== CHATBOT REQUIREMENTS CHECK ===\n")

for package, description in packages.items():
    if importlib.util.find_spec(package):
        print(f"✅ {package:<25} INSTALLED | {description}")
    else:
        print(f"❌ {package:<25} NOT INSTALLED | {description}")


# In[30]:


# Stage 7.2 - Add Conversation Memory Foundation to app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# -----------------------------------------------
# Add conversation memory after Streamlit config
# -----------------------------------------------

old_code = '''st.set_page_config(
    page_title="AI Surface Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)'''

new_code = '''st.set_page_config(
    page_title="AI Surface Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)

# =================================================
# CONVERSATION MEMORY
# =================================================

# Store the conversation history across Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the latest computer vision analysis
# so the chatbot can answer questions about the uploaded image
if "latest_detection_summary" not in st.session_state:
    st.session_state.latest_detection_summary = (
        "No image has been analyzed yet."
    )

# Store the latest RAG context
if "latest_retrieved_context" not in st.session_state:
    st.session_state.latest_retrieved_context = (
        "No contextual knowledge has been retrieved yet."
    )
'''

if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "Conversation memory foundation "
        "added to app.py successfully!"
    )

else:

    print(
        "Could not find the Streamlit configuration block. "
        "No changes were made."
    )


# In[31]:


# Stage 7.3 - Store YOLO and RAG Results in Conversation Memory

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# -------------------------------------------------
# 1. Save latest YOLO detection summary
# -------------------------------------------------

old_detection_code = '''    st.subheader("Exact Detection Summary")'''

new_detection_code = '''    # Save latest YOLO result in session memory
    st.session_state.latest_detection_summary = detection_summary

    st.subheader("Exact Detection Summary")'''


if old_detection_code in app_code:
    app_code = app_code.replace(
        old_detection_code,
        new_detection_code,
        1
    )

    detection_added = True

else:
    detection_added = False


# -------------------------------------------------
# 2. Save latest retrieved RAG context
# -------------------------------------------------

old_rag_code = '''    # Display retrieved RAG knowledge
    st.subheader(
        "Retrieved Knowledge (RAG)"
    )'''

new_rag_code = '''    # Save latest RAG context in session memory
    st.session_state.latest_retrieved_context = retrieved_context

    # Display retrieved RAG knowledge
    st.subheader(
        "Retrieved Knowledge (RAG)"
    )'''


if old_rag_code in app_code:
    app_code = app_code.replace(
        old_rag_code,
        new_rag_code,
        1
    )

    rag_added = True

else:
    rag_added = False


# -------------------------------------------------
# 3. Save updated app.py
# -------------------------------------------------

if detection_added and rag_added:

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "YOLO detection summary and RAG context "
        "are now stored in conversation memory!"
    )

else:

    print("WARNING: One or more code sections were not found.")

    print(
        "Detection memory added:",
        detection_added
    )

    print(
        "RAG memory added:",
        rag_added
    )


# In[32]:


# Stage 7.4 - Add Multi-turn Chat Interface to Streamlit App

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


chatbot_code = r'''

# =================================================
# INTERACTIVE CONVERSATIONAL ASSISTANT
# =================================================

st.divider()

st.header("💬 Chat with AI Assistant")

st.write(
    "Ask questions about the detected defects, "
    "inspection results, or retrieved knowledge."
)


# -------------------------------------------------
# Display previous conversation
# -------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# -------------------------------------------------
# User chat input
# -------------------------------------------------

user_question = st.chat_input(
    "Ask a question about the inspection..."
)


if user_question:

    # Save user message to conversation memory
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # ---------------------------------------------
    # Build conversation history
    # ---------------------------------------------

    conversation_history = ""

    for message in st.session_state.messages[:-1]:

        conversation_history += (
            f'{message["role"].upper()}: '
            f'{message["content"]}\n'
        )


    # ---------------------------------------------
    # Build grounded chatbot prompt
    # ---------------------------------------------

    chat_prompt = f"""
You are an AI assistant for a surface defect inspection system.

LATEST COMPUTER VISION ANALYSIS:
{st.session_state.latest_detection_summary}

RETRIEVED KNOWLEDGE:
{st.session_state.latest_retrieved_context}

PREVIOUS CONVERSATION:
{conversation_history}

CURRENT USER QUESTION:
{user_question}

INSTRUCTIONS:

- Answer the user's current question clearly and concisely.
- Use the computer vision results when the question refers to the uploaded image.
- Do not invent or change detection counts or confidence scores.
- Use the retrieved knowledge as supporting context.
- Consider the previous conversation when answering follow-up questions.
- Do not make a definitive structural safety diagnosis.
- If the available information is insufficient, clearly say so.
"""


    # ---------------------------------------------
    # Generate chatbot response using Ollama
    # ---------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Generating response..."
        ):

            chat_response = ollama.chat(
                model="qwen2.5:3b",
                messages=[
                    {
                        "role": "user",
                        "content": chat_prompt
                    }
                ]
            )

            assistant_answer = (
                chat_response["message"]["content"]
            )

            st.markdown(
                assistant_answer
            )


    # ---------------------------------------------
    # Save assistant response to memory
    # ---------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_answer
        }
    )

'''


# Add chatbot section at the end of app.py
if "INTERACTIVE CONVERSATIONAL ASSISTANT" not in app_code:

    app_code = app_code + chatbot_code

    with open(
        "app.py",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            app_code
        )

    print(
        "Multi-turn conversational chatbot "
        "added to app.py successfully!"
    )

else:

    print(
        "Chatbot section already exists. "
        "No duplicate code was added."
    )


# In[33]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)
    print("SUCCESS: app.py syntax check passed!")
    print("Chatbot integration is ready for testing.")

except py_compile.PyCompileError as e:
    print("ERROR: Syntax error found in app.py")
    print(e)


# In[34]:


# Stage 7.5.1 - Add Fresh FAISS Retrieval Function for Chat Questions

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


fresh_rag_function = r'''

# =================================================
# FRESH RAG RETRIEVAL FOR CHAT QUESTIONS
# =================================================

def retrieve_chat_knowledge(user_question, top_k=3):

    # Convert the user's current question into an embedding
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=user_question
    )

    query_embedding = np.array(
        [response["embedding"]]
    ).astype("float32")


    # Search the FAISS vector database
    distances, indices = index.search(
        query_embedding,
        top_k
    )


    # Store retrieved knowledge
    retrieved_chat_documents = []

    for rank, doc_index in enumerate(indices[0]):

        if doc_index >= 0:

            retrieved_chat_documents.append(
                knowledge_documents[doc_index]
            )


    # Combine retrieved documents as context
    retrieved_chat_context = "\n\n".join(
        retrieved_chat_documents
    )


    return (
        retrieved_chat_context,
        retrieved_chat_documents
    )

'''


# Insert the function before the interactive chatbot section
marker = "# INTERACTIVE CONVERSATIONAL ASSISTANT"

if "def retrieve_chat_knowledge(" not in app_code:

    if marker in app_code:

        app_code = app_code.replace(
            marker,
            fresh_rag_function + "\n" + marker,
            1
        )

        with open(
            "app.py",
            "w",
            encoding="utf-8"
        ) as file:

            file.write(app_code)

        print(
            "Fresh FAISS retrieval function "
            "added to app.py successfully!"
        )

    else:

        print(
            "ERROR: Chatbot section marker "
            "was not found in app.py."
        )

else:

    print(
        "Fresh FAISS retrieval function "
        "already exists."
    )


# In[35]:


# Stage 7.5.2 - Connect Fresh FAISS Retrieval to Chatbot

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''    # ---------------------------------------------
    # Build conversation history
    # ---------------------------------------------

    conversation_history = ""
'''


new_code = '''    # ---------------------------------------------
    # Fresh RAG retrieval for the current question
    # ---------------------------------------------

    fresh_chat_context, fresh_chat_documents = retrieve_chat_knowledge(
        user_question,
        top_k=3
    )


    # ---------------------------------------------
    # Build conversation history
    # ---------------------------------------------

    conversation_history = ""
'''


if "fresh_chat_context, fresh_chat_documents = retrieve_chat_knowledge(" not in app_code:

    if old_code in app_code:

        app_code = app_code.replace(
            old_code,
            new_code,
            1
        )

        # Replace the old stored RAG context in the chatbot prompt
        app_code = app_code.replace(
            '''RETRIEVED KNOWLEDGE:
{st.session_state.latest_retrieved_context}

PREVIOUS CONVERSATION:''',
            '''FRESHLY RETRIEVED KNOWLEDGE FOR CURRENT QUESTION:
{fresh_chat_context}

PREVIOUS CONVERSATION:''',
            1
        )

        with open(
            "app.py",
            "w",
            encoding="utf-8"
        ) as file:
            file.write(app_code)

        print(
            "Fresh FAISS retrieval connected "
            "to chatbot successfully!"
        )

    else:

        print(
            "ERROR: Could not find the chatbot "
            "conversation history section."
        )

else:

    print(
        "Fresh FAISS retrieval is already "
        "connected to the chatbot."
    )


# In[36]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)

    print("SUCCESS: app.py syntax check passed!")
    print("Fresh RAG chatbot integration is ready for testing.")

except py_compile.PyCompileError as e:

    print("ERROR: Syntax error found in app.py")
    print(e)


# In[37]:


# Check FAISS index variable names used in app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

print("=== FAISS INDEX REFERENCES IN app.py ===\n")

for line_number, line in enumerate(app_code.splitlines(), start=1):
    if "faiss" in line.lower() or "index" in line.lower():
        print(f"{line_number}: {line}")


# In[38]:


# Fix incorrect FAISS index variable in fresh chatbot retrieval

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

old_code = '''distances, indices = index.search(
        query_embedding,
        top_k
    )'''

new_code = '''distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )'''

if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print("SUCCESS: Chatbot FAISS index variable fixed!")
    print("Changed: index.search() -> faiss_index.search()")

else:
    print("The target code was not found.")
    print("Please do not make any further changes.")


# In[39]:


# Check app.py syntax after FAISS variable fix

import ast

try:
    with open("app.py", "r", encoding="utf-8") as file:
        app_code = file.read()

    ast.parse(app_code)

    print("SUCCESS: app.py syntax check passed!")
    print("FAISS chatbot fix is ready for testing.")

except SyntaxError as e:
    print("SYNTAX ERROR:")
    print(e)


# In[40]:


# Stage 7.6.1 - Add Source Labels to Fresh RAG Retrieval

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''    # Store retrieved knowledge
    retrieved_chat_documents = []

    for rank, doc_index in enumerate(indices[0]):

        if doc_index >= 0:

            retrieved_chat_documents.append(
                knowledge_documents[doc_index]
            )


    # Combine retrieved documents as context
    retrieved_chat_context = "\\n\\n".join(
        retrieved_chat_documents
    )'''


new_code = '''    # Store retrieved knowledge with source labels
    retrieved_chat_documents = []

    for rank, doc_index in enumerate(indices[0]):

        if doc_index >= 0:

            source_label = f"Knowledge Source {doc_index + 1}"

            retrieved_chat_documents.append({
                "source": source_label,
                "content": knowledge_documents[doc_index]
            })


    # Combine retrieved document content as context
    retrieved_chat_context = "\\n\\n".join(
        document["content"]
        for document in retrieved_chat_documents
    )'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print("Source labels added to fresh RAG retrieval successfully!")

else:

    print("ERROR: Target retrieval code was not found.")
    print("No changes were made.")


# In[41]:


# Stage 7.6.2 - Display Retrieved Sources Below Chatbot Answers

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# Find the point where the assistant response is stored
old_code = '''    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": assistant_response
    })'''


new_code = '''    # Prepare source labels used for this answer
    answer_sources = [
        document["source"]
        for document in fresh_chat_documents
    ]

    # Store assistant response together with retrieved sources
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": assistant_response,
        "sources": answer_sources
    })'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "Retrieved source labels stored with "
        "chatbot answers successfully!"
    )

else:

    print("ERROR: Assistant message storage code was not found.")
    print("No changes were made.")


# In[42]:


# Find chatbot message storage code in app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

print("=== CHAT MESSAGE REFERENCES ===\n")

for line_number, line in enumerate(app_code.splitlines(), start=1):
    if (
        "chat_messages" in line
        or "assistant_response" in line
        or '"role"' in line
        or "'role'" in line
    ):
        print(f"{line_number}: {line}")


# In[43]:


# Show chatbot assistant message storage section

with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== ASSISTANT MESSAGE STORAGE SECTION ===\n")

# Display lines 610 to 635
for line_number in range(610, min(636, len(app_lines) + 1)):
    print(
        f"{line_number}: "
        f"{app_lines[line_number - 1]}",
        end=""
    )


# In[44]:


# Stage 7.6.2 - Store Retrieved Sources with Assistant Answers

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_answer
        }
    )'''


new_code = '''    # Prepare source labels used for this answer
    answer_sources = [
        document["source"]
        for document in fresh_chat_documents
    ]

    # Save assistant response and retrieved sources to memory
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_answer,
            "sources": answer_sources
        }
    )'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "Retrieved source labels stored with "
        "chatbot answers successfully!"
    )

else:

    print("ERROR: Assistant message storage code was not found.")
    print("No changes were made.")


# In[45]:


# Stage 7.6.3 - Display Retrieved Sources in Chat History

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )'''


new_code = '''for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # Display retrieved RAG sources for assistant answers
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources used for this answer"
            ):

                for source in message["sources"]:

                    st.write(
                        f"- {source}"
                    )'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "Retrieved sources display added "
        "to chatbot UI successfully!"
    )

else:

    print("ERROR: Chat history display code was not found.")
    print("No changes were made.")
    


# In[46]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)

    print("SUCCESS: app.py syntax check passed!")
    print("Chatbot source display is ready for testing.")

except py_compile.PyCompileError as e:

    print("ERROR: Syntax error found in app.py")
    print(e)


# In[47]:


# Stage 7.6.4 - Display Sources Immediately Below Current Answer

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''            st.markdown(
                assistant_answer
            )'''


new_code = '''            st.markdown(
                assistant_answer
            )

            # Display RAG sources immediately below current answer
            if fresh_chat_documents:

                with st.expander(
                    "📚 Sources used for this answer"
                ):

                    for document in fresh_chat_documents:

                        st.write(
                            f"- {document['source']}"
                        )'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "Current chatbot answer source display "
        "added successfully!"
    )

else:

    print("ERROR: Current assistant answer display code was not found.")
    print("No changes were made.")


# In[48]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)
    print("SUCCESS: app.py syntax check passed!")
    print("Current chatbot source display is ready for testing.")

except py_compile.PyCompileError as e:
    print("ERROR: Syntax error found in app.py")
    print(e)


# In[49]:


with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== KNOWLEDGE DOCUMENT SECTION ===\n")

for line_number in range(40, min(95, len(app_lines) + 1)):
    print(
        f"{line_number}: "
        f"{app_lines[line_number - 1]}",
        end=""
    )


# In[50]:


# Stage 7.7 - Add Meaningful Source Titles

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''knowledge_documents = ['''


new_code = '''knowledge_source_titles = [
    "Crack Detection and Inspection Guide",
    "Hole Detection and Assessment Guide",
    "Normal Surface and Maintenance Guide",
    "Structural Inspection Recommendations"
]

knowledge_documents = ['''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    # Replace generic source-label creation
    old_source = '''"source": f"Knowledge Source {doc_index + 1}"'''

    new_source = '''"source": knowledge_source_titles[doc_index]'''

    if old_source in app_code:

        app_code = app_code.replace(
            old_source,
            new_source
        )

        with open("app.py", "w", encoding="utf-8") as file:
            file.write(app_code)

        print("Meaningful RAG source titles added successfully!")

    else:

        print("ERROR: Generic source label code was not found.")
        print("No changes were saved.")

else:

    print("ERROR: knowledge_documents section was not found.")
    print("No changes were made.")


# In[51]:


with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== SOURCE LABEL REFERENCES ===\n")

for line_number, line in enumerate(app_lines, start=1):
    if (
        "Knowledge Source" in line
        or '"source"' in line
        or "'source'" in line
    ):
        print(
            f"{line_number}: {line}",
            end=""
        )


# In[52]:


# Stage 7.7 - Add Meaningful RAG Source Titles

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


# Add titles before knowledge_documents
old_documents_start = '''knowledge_documents = ['''

new_documents_start = '''knowledge_source_titles = [
    "Crack Detection and Inspection Guide",
    "Hole Detection and Assessment Guide",
    "Normal Surface and Maintenance Guide",
    "Structural Inspection Recommendations"
]

knowledge_documents = ['''


# Replace generic source label
old_source_label = '''source_label = f"Knowledge Source {doc_index + 1}"'''

new_source_label = '''source_label = knowledge_source_titles[doc_index]'''


# Check both sections before making changes
if (
    old_documents_start in app_code
    and old_source_label in app_code
):

    app_code = app_code.replace(
        old_documents_start,
        new_documents_start,
        1
    )

    app_code = app_code.replace(
        old_source_label,
        new_source_label,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print("Meaningful RAG source titles added successfully!")

else:

    print("ERROR: Required code section was not found.")
    print("No changes were made.")


# In[53]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)

    print("SUCCESS: app.py syntax check passed!")
    print("Meaningful RAG source titles are ready for testing.")

except py_compile.PyCompileError as e:

    print("ERROR: Syntax error found in app.py")
    print(e)


# In[54]:


with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== CURRENT RAG KNOWLEDGE BASE ===\n")

for line_number in range(50, min(110, len(app_lines) + 1)):
    print(
        f"{line_number}: "
        f"{app_lines[line_number - 1]}",
        end=""
    )


# In[55]:


# Stage 7.8.1 - Expand RAG Knowledge Base to 6 Documents

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_titles = '''knowledge_source_titles = [
    "Crack Detection and Inspection Guide",
    "Hole Detection and Assessment Guide",
    "Normal Surface and Maintenance Guide",
    "Structural Inspection Recommendations"
]'''


new_titles = '''knowledge_source_titles = [
    "Crack Detection and Inspection Guide",
    "Hole Detection and Assessment Guide",
    "Normal Surface and Maintenance Guide",
    "Structural Inspection Recommendations",
    "Defect Confidence Score Interpretation Guide",
    "Defect Monitoring and Follow-up Guide"
]'''


old_last_document = '''    Inspection Recommendation:
    Computer vision detections should be treated as automated inspection support.
    High-confidence detections can be prioritized for further inspection.
    Lower-confidence detections may require manual visual verification.
    Final decisions regarding structural condition or repair should be made
    after appropriate professional inspection.
    """
]'''


new_last_document = '''    Inspection Recommendation:
    Computer vision detections should be treated as automated inspection support.
    High-confidence detections can be prioritized for further inspection.
    Lower-confidence detections may require manual visual verification.
    Final decisions regarding structural condition or repair should be made
    after appropriate professional inspection.
    """,

    """
    Defect Confidence Score Interpretation:
    A confidence score represents the computer vision model's certainty about
    a detected defect. Higher-confidence detections indicate stronger model
    certainty and can be prioritized for inspection. Lower-confidence detections
    should not automatically be ignored, because they may represent uncertain
    defects that require manual visual verification. Confidence scores support
    inspection prioritization but should not be treated as a final assessment
    of structural severity.
    """,

    """
    Defect Monitoring and Follow-up:
    Detected cracks and holes should be documented and monitored when appropriate.
    Follow-up inspection can be used to identify changes in crack length, width,
    hole size, or other visible characteristics over time. Repeated observations
    can help determine whether a defect appears stable or is progressing.
    Significant changes or uncertain conditions should be referred for appropriate
    professional inspection before repair or structural decisions are made.
    """
]'''


if (
    old_titles in app_code
    and old_last_document in app_code
):

    app_code = app_code.replace(
        old_titles,
        new_titles,
        1
    )

    app_code = app_code.replace(
        old_last_document,
        new_last_document,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print("RAG knowledge base expanded successfully!")
    print("Total knowledge documents: 6")
    print("Total source titles: 6")

else:

    print("ERROR: Required knowledge base section was not found.")
    print("No changes were made.")


# In[58]:


# Verify updated knowledge base directly from app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== UPDATED RAG KNOWLEDGE BASE IN app.py ===\n")

for line_number in range(50, min(125, len(app_lines) + 1)):
    print(
        f"{line_number}: "
        f"{app_lines[line_number - 1]}",
        end=""
    )


# In[ ]:





# In[59]:


import importlib.util

print("=== LANGCHAIN INTEGRATION CHECK ===\n")

modules = {
    "langchain": "LangChain Core Framework",
    "langchain_community": "LangChain Community Integrations"
}

for module, description in modules.items():
    if importlib.util.find_spec(module):
        print(f"✅ {module:<25} INSTALLED | {description}")
    else:
        print(f"❌ {module:<25} NOT INSTALLED | {description}")

print("\n=== CHECK COMPLETE ===")


# In[60]:


app_file = "app.py"

print("=== LANGCHAIN REFERENCES IN app.py ===\n")

with open(app_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

found = False

for line_number, line in enumerate(lines, start=1):
    if "langchain" in line.lower():
        print(f"{line_number}: {line.rstrip()}")
        found = True

if not found:
    print("❌ No LangChain references found in app.py")


# In[62]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)
    print("SUCCESS: app.py syntax check passed!")
    print("LangChain imports added successfully!")
except Exception as e:
    print("ERROR:", e)


# In[63]:


# Stage 8.2 Step 2
# Add LangChain Document objects to the actual RAG knowledge base

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


marker = '''# =================================================
# CREATE FAISS VECTOR INDEX
# ================================================='''


langchain_document_code = '''# =================================================
# LANGCHAIN DOCUMENT STRUCTURE
# =================================================

# Convert the RAG knowledge base into LangChain Document objects.
# Each document contains the knowledge text and its source metadata.
langchain_documents = [
    Document(
        page_content=document,
        metadata={"source": knowledge_source_titles[i]}
    )
    for i, document in enumerate(knowledge_documents)
]


'''


if "langchain_documents = [" not in app_code:

    if marker in app_code:

        app_code = app_code.replace(
            marker,
            langchain_document_code + marker,
            1
        )

        with open("app.py", "w", encoding="utf-8") as file:
            file.write(app_code)

        print(
            "SUCCESS: LangChain Document objects "
            "added to the RAG pipeline!"
        )

    else:

        print(
            "ERROR: FAISS index section marker was not found."
        )

else:

    print(
        "LangChain Document objects already exist."
    )


# In[64]:


# Stage 8.2 Step 3
# Use LangChain Document objects in actual chatbot RAG retrieval

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''            source_label = knowledge_source_titles[doc_index]

            retrieved_chat_documents.append({
                "source": source_label,
                "content": knowledge_documents[doc_index]
            })'''


new_code = '''            # Retrieve the corresponding LangChain Document
            retrieved_document = langchain_documents[doc_index]

            retrieved_chat_documents.append({
                "source": retrieved_document.metadata["source"],
                "content": retrieved_document.page_content
            })'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "SUCCESS: LangChain Document objects are now "
        "used in the actual RAG retrieval pipeline!"
    )

else:

    print("ERROR: Chat retrieval document section was not found.")
    print("No changes were made.")


# In[65]:


# Find the current chatbot prompt section in app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_lines = file.readlines()

print("=== CHATBOT PROMPT SECTION ===\n")

for line_number, line in enumerate(app_lines, start=1):
    if (
        "chat_prompt" in line
        or "LATEST COMPUTER VISION ANALYSIS" in line
        or "FRESHLY RETRIEVED KNOWLEDGE" in line
        or "CURRENT USER QUESTION" in line
    ):
        start = max(1, line_number - 5)
        end = min(len(app_lines), line_number + 30)

        for i in range(start, end + 1):
            print(
                f"{i}: {app_lines[i - 1]}",
                end=""
            )

        break


# In[66]:


# Stage 8.2 Step 4.2
# Replace manual chatbot prompt with LangChain PromptTemplate

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()


old_code = '''    chat_prompt = f"""
You are an AI assistant for a surface defect inspection system.

LATEST COMPUTER VISION ANALYSIS:
{st.session_state.latest_detection_summary}

FRESHLY RETRIEVED KNOWLEDGE FOR CURRENT QUESTION:
{fresh_chat_context}

PREVIOUS CONVERSATION:
{conversation_history}

CURRENT USER QUESTION:
{user_question}

INSTRUCTIONS:

- Answer the user's current question clearly and concisely.
- Use the computer vision results when the question refers to the uploaded image.
- Do not invent or change detection counts or confidence scores.
- Use the retrieved knowledge as supporting context.
- Consider the previous conversation when answering follow-up questions.
- Do not make a definitive structural safety diagnosis.
- If the available information is insufficient, clearly say so.
"""'''


new_code = '''    # Build the grounded chatbot prompt using LangChain PromptTemplate
    chatbot_prompt_template = PromptTemplate(
        input_variables=[
            "detection_summary",
            "retrieved_context",
            "conversation_history",
            "user_question"
        ],
        template="""
You are an AI assistant for a surface defect inspection system.

LATEST COMPUTER VISION ANALYSIS:
{detection_summary}

FRESHLY RETRIEVED KNOWLEDGE FOR CURRENT QUESTION:
{retrieved_context}

PREVIOUS CONVERSATION:
{conversation_history}

CURRENT USER QUESTION:
{user_question}

INSTRUCTIONS:

- Answer the user's current question clearly and concisely.
- Use the computer vision results when the question refers to the uploaded image.
- Do not invent or change detection counts or confidence scores.
- Use the retrieved knowledge as supporting context.
- Consider the previous conversation when answering follow-up questions.
- Do not make a definitive structural safety diagnosis.
- If the available information is insufficient, clearly say so.
"""
    )

    # Format the final prompt through LangChain
    chat_prompt = chatbot_prompt_template.format(
        detection_summary=st.session_state.latest_detection_summary,
        retrieved_context=fresh_chat_context,
        conversation_history=conversation_history,
        user_question=user_question
    )'''


if old_code in app_code:

    app_code = app_code.replace(
        old_code,
        new_code,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print(
        "SUCCESS: LangChain PromptTemplate is now "
        "used in the actual chatbot pipeline!"
    )

else:

    print("ERROR: Existing chatbot prompt section was not found.")
    print("No changes were made.")


# In[67]:


import py_compile

try:
    py_compile.compile("app.py", doraise=True)

    print("SUCCESS: app.py syntax check passed!")
    print("LangChain Document + PromptTemplate integration is ready for testing.")

except py_compile.PyCompileError as e:

    print("ERROR: Syntax error found in app.py")
    print(e)


# In[68]:


with open("app.py", "r", encoding="utf-8") as file:
    lines = file.readlines()

print("=== FIRST 35 LINES OF app.py ===\n")

for i, line in enumerate(lines[:35], start=1):
    print(f"{i}: {line}", end="")


# In[69]:


# Add required LangChain imports to app.py

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

import_marker = "import numpy as np"

langchain_imports = """import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate"""

if "from langchain_core.documents import Document" not in app_code:
    app_code = app_code.replace(
        import_marker,
        langchain_imports,
        1
    )

    with open("app.py", "w", encoding="utf-8") as file:
        file.write(app_code)

    print("SUCCESS: Required LangChain imports added to app.py!")
else:
    print("LangChain Document import already exists.")


# In[70]:


import py_compile

print("=== LANGCHAIN IMPORT VERIFICATION ===\n")

with open("app.py", "r", encoding="utf-8") as file:
    app_code = file.read()

checks = [
    "from langchain_core.documents import Document",
    "from langchain_core.prompts import PromptTemplate"
]

for item in checks:
    if item in app_code:
        print(f"✅ FOUND: {item}")
    else:
        print(f"❌ MISSING: {item}")

print("\n=== SYNTAX CHECK ===")

try:
    py_compile.compile("app.py", doraise=True)
    print("✅ SUCCESS: app.py syntax check passed!")
except Exception as e:
    print("❌ ERROR:", e)


# In[71]:


# =================================================
# STAGE 8.2 - FINAL LANGCHAIN INTEGRATION VERIFICATION
# =================================================

with open("app.py", "r", encoding="utf-8") as file:
    lines = file.readlines()

checks = {
    "Document Import":
        "from langchain_core.documents import Document",

    "PromptTemplate Import":
        "from langchain_core.prompts import PromptTemplate",

    "LangChain Document Usage":
        "Document(",

    "Document Page Content":
        "page_content=",

    "Document Metadata":
        "metadata=",

    "PromptTemplate Usage":
        "PromptTemplate(",

    "PromptTemplate Formatting":
        "chatbot_prompt_template.format("
}

print("=== STAGE 8.2 LANGCHAIN FINAL VERIFICATION ===\n")

all_passed = True

for check_name, search_text in checks.items():

    matches = []

    for line_number, line in enumerate(lines, start=1):
        if search_text in line:
            matches.append(line_number)

    if matches:
        print(
            f"✅ {check_name}: FOUND "
            f"(Line(s): {', '.join(map(str, matches))})"
        )
    else:
        print(f"❌ {check_name}: NOT FOUND")
        all_passed = False


print("\n=== FINAL RESULT ===")

if all_passed:
    print("✅ STAGE 8.2 CODE VERIFICATION PASSED")
    print("LangChain Document and PromptTemplate are present in app.py.")
else:
    print("❌ STAGE 8.2 CODE VERIFICATION FAILED")
    print("One or more required LangChain components are missing.")


# In[3]:


from pathlib import Path
import shutil
import py_compile
from datetime import datetime

# ============================================================
# APP.PY PATH
# ============================================================

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

# ============================================================
# 1. CHECK APP.PY
# ============================================================

if not app_path.exists():
    raise FileNotFoundError(f"app.py not found:\n{app_path}")

print("✅ app.py found")


# ============================================================
# 2. CREATE BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = app_path.with_name(
    f"app_before_model_selector_{timestamp}.py"
)

shutil.copy2(app_path, backup_path)

print("✅ Safety backup created:")
print(backup_path)


# ============================================================
# 3. READ APP.PY
# ============================================================

code = app_path.read_text(encoding="utf-8")


# ============================================================
# 4. MODEL SELECTOR CODE
# ============================================================

selector_code = '''

# =================================================
# OLLAMA AI MODEL SELECTION
# =================================================

AVAILABLE_LLM_MODELS = {
    "Qwen 2.5 3B (Fast / Default)": "qwen2.5:3b",
    "Llama 3.2 1B (Lightweight)": "llama3.2:1b",
    "Llama 3.1 8B (Higher Quality)": "llama3.1:8b",
    "Gemma 3 12B (Most Powerful)": "gemma3:12b",
}

selected_model_label = st.sidebar.selectbox(
    "🤖 Select AI Model",
    options=list(AVAILABLE_LLM_MODELS.keys()),
    index=0,
    help="Choose the Ollama language model used for AI inspection reports and chatbot responses."
)

selected_llm_model = AVAILABLE_LLM_MODELS[selected_model_label]

st.sidebar.caption(
    f"Active model: {selected_llm_model}"
)

'''


# ============================================================
# 5. FIND STREAMLIT CONFIG
# ============================================================

marker = '''st.set_page_config(
    page_title="AI Surface Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)'''


# ============================================================
# 6. ADD MODEL SELECTOR
# ============================================================

if "AVAILABLE_LLM_MODELS =" not in code:

    if marker not in code:
        raise RuntimeError(
            "Could not find st.set_page_config block."
        )

    code = code.replace(
        marker,
        marker + selector_code,
        1
    )

    print("✅ Model selector added")

else:

    print("ℹ️ Model selector already exists")


# ============================================================
# 7. REPLACE ONLY CHAT LLM MODEL REFERENCES
# ============================================================

old_model = 'model="qwen2.5:3b",'
new_model = 'model=selected_llm_model,'

count = code.count(old_model)

print(
    f"🔎 Found {count} hard-coded qwen2.5:3b reference(s)"
)

if count > 0:

    code = code.replace(
        old_model,
        new_model
    )

    print(
        f"✅ Replaced {count} LLM model reference(s)"
    )

elif "model=selected_llm_model," in code:

    print(
        "ℹ️ LLM references already use selected_llm_model"
    )

else:

    raise RuntimeError(
        "Could not find the expected qwen2.5:3b model references."
    )


# ============================================================
# 8. SAVE UPDATED APP.PY
# ============================================================

app_path.write_text(
    code,
    encoding="utf-8"
)

print("✅ Updated app.py saved")


# ============================================================
# 9. SYNTAX CHECK
# ============================================================

try:

    py_compile.compile(
        str(app_path),
        doraise=True
    )

except Exception as e:

    print("\n❌ app.py syntax check FAILED")
    print(e)

    print("\nRestoring original app.py...")

    shutil.copy2(
        backup_path,
        app_path
    )

    print("✅ Original app.py restored")

    raise


# ============================================================
# 10. FINAL VERIFICATION
# ============================================================

final_code = app_path.read_text(
    encoding="utf-8"
)

checks = {
    "Model selector":
        "AVAILABLE_LLM_MODELS =" in final_code,

    "Selected model variable":
        "selected_llm_model =" in final_code,

    "Dynamic Ollama model usage":
        final_code.count(
            "model=selected_llm_model,"
        ) >= 2,

    "RAG embedding model preserved":
        'model="nomic-embed-text:latest"' in final_code,
}


print("\n========================================")
print("FINAL VERIFICATION")
print("========================================")

all_good = True

for name, result in checks.items():

    if result:
        print(f"✅ {name}: OK")
    else:
        print(f"❌ {name}: FAILED")
        all_good = False


if all_good:

    print("\n========================================")
    print("🎉 SUCCESS")
    print("========================================")

    print(
        "Model selector successfully integrated into app.py!"
    )

    print("\nAvailable models:")
    print(" • Qwen 2.5 3B")
    print(" • Llama 3.2 1B")
    print(" • Llama 3.1 8B")
    print(" • Gemma 3 12B")

    print("\nEmbedding model unchanged:")
    print(" • nomic-embed-text:latest")

    print("\nBackup:")
    print(backup_path)

else:

    print("\n❌ Verification failed.")

    print(
        "Restoring original app.py for safety..."
    )

    shutil.copy2(
        backup_path,
        app_path
    )

    print("✅ Original app.py restored.")


# In[3]:


from pathlib import Path

project_folder = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

print("Searching for YOLO model files...\n")

pt_files = list(project_folder.rglob("*.pt"))

if pt_files:
    print(f"Found {len(pt_files)} .pt file(s):\n")
    
    for file in pt_files:
        print(file)
else:
    print("No .pt files found inside the project folder.")


# In[4]:


from ultralytics import YOLO

model_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"

model = YOLO(model_path)

print("YOLO Model Loaded Successfully!")

print("\nYOLO Model Classes:")
print(model.names)

print("\nNumber of Classes:")
print(len(model.names))


# In[5]:


from pathlib import Path

project_folder = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

yaml_files = list(project_folder.rglob("*.yaml"))

print("YAML files found:\n")

for file in yaml_files:
    print(file)


# In[6]:


from pathlib import Path

yaml_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset\data.yaml"
)

print("===== DATA.YAML CONTENT =====\n")

with open(yaml_path, "r") as file:
    print(file.read())


# In[7]:


from pathlib import Path

dataset = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\Crack_Hole_Normal_Dataset"
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "val", "test"]:
    folder = dataset / "images" / split

    images = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

    print(f"{split.upper()} images: {len(images)}")

print("\nTOTAL images:",
      sum(
          len([
              f for f in (dataset / "images" / split).rglob("*")
              if f.is_file() and f.suffix.lower() in extensions
          ])
          for split in ["train", "val", "test"]
      ))


# In[9]:


from pathlib import Path
import csv

results_csv = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
    r"\runs\detect\NHPT_Project_v2\Fine_Tuned-2\results.csv"
)

if not results_csv.exists():
    print("❌ results.csv not found:")
    print(results_csv)

else:
    print("✅ Training results found!")
    print("File:", results_csv)

    with open(results_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        # Remove extra spaces from Ultralytics column names
        rows = []
        for row in reader:
            clean_row = {
                key.strip(): value.strip()
                for key, value in row.items()
            }
            rows.append(clean_row)

    if not rows:
        print("❌ results.csv is empty.")

    else:
        final = rows[-1]

        print("\n===== AVAILABLE COLUMNS =====")
        for key in final.keys():
            print(key)

        print("\n===== FINAL EPOCH METRICS =====")

        wanted = [
            "epoch",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)"
        ]

        for key in wanted:
            if key in final:
                print(f"{key}: {final[key]}")

        # Calculate F1
        p_key = "metrics/precision(B)"
        r_key = "metrics/recall(B)"

        if p_key in final and r_key in final:
            precision = float(final[p_key])
            recall = float(final[r_key])

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                print(f"F1 Score: {f1:.4f}")


# In[10]:


from pathlib import Path
import time
from statistics import mean

test_folder = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
    r"\Crack_Hole_Normal_Dataset\images\test"
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

test_images = [
    f for f in test_folder.rglob("*")
    if f.is_file() and f.suffix.lower() in extensions
]

# Use first 20 test images
sample_images = test_images[:20]

print(f"Test images found: {len(test_images)}")
print(f"Images used for timing: {len(sample_images)}")

inference_times = []

# Warm-up prediction
if sample_images:
    model.predict(
        source=str(sample_images[0]),
        verbose=False
    )

# Measure inference
for image_path in sample_images:
    start = time.perf_counter()

    model.predict(
        source=str(image_path),
        verbose=False
    )

    end = time.perf_counter()

    inference_times.append(
        (end - start) * 1000
    )

if inference_times:
    avg_time = mean(inference_times)

    print("\n===== INFERENCE TIME RESULTS =====")
    print(f"Average end-to-end prediction time: {avg_time:.2f} ms/image")
    print(f"Fastest prediction: {min(inference_times):.2f} ms")
    print(f"Slowest prediction: {max(inference_times):.2f} ms")
    print(f"Approx. throughput: {1000 / avg_time:.2f} images/second")
else:
    print("❌ No test images available.")


# In[11]:


from pathlib import Path

test_folder = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
    r"\Crack_Hole_Normal_Dataset\images\test"
)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

test_images = [
    f for f in test_folder.rglob("*")
    if f.is_file() and f.suffix.lower() in extensions
]

print(f"Test images found: {len(test_images)}")
print("Running existing YOLO model on test images...")
print("No training is being performed.\n")

area_ratios = {
    "crack": [],
    "hole": []
}

detections_count = {
    "crack": 0,
    "hole": 0
}

# Use all 400 test images
for image_path in test_images:

    results = model.predict(
        source=str(image_path),
        verbose=False
    )

    for result in results:

        img_h, img_w = result.orig_shape
        image_area = img_w * img_h

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id].lower()

            # We only need actual defect classes
            if class_name not in ["crack", "hole"]:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)

            box_area = box_width * box_height

            ratio = (box_area / image_area) * 100

            area_ratios[class_name].append(ratio)
            detections_count[class_name] += 1


print("===== DEFECT BOUNDING-BOX AREA ANALYSIS =====")

for defect in ["crack", "hole"]:

    values = sorted(area_ratios[defect])

    print(f"\n{defect.upper()}")
    print(f"Detections: {detections_count[defect]}")

    if values:

        n = len(values)

        median = values[n // 2]

        q1 = values[int((n - 1) * 0.25)]
        q3 = values[int((n - 1) * 0.75)]

        print(f"Minimum area ratio: {min(values):.4f}%")
        print(f"Q1 area ratio:      {q1:.4f}%")
        print(f"Median area ratio:  {median:.4f}%")
        print(f"Q3 area ratio:      {q3:.4f}%")
        print(f"Maximum area ratio: {max(values):.4f}%")

    else:
        print("No detections found.")


# In[12]:


from pathlib import Path

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

lines = app_path.read_text(encoding="utf-8").splitlines()

print("===== DETECTION SUMMARY AREA =====\n")

for i, line in enumerate(lines, start=1):
    if "summary_lines" in line or "detection_summary" in line:
        start = max(1, i - 8)
        end = min(len(lines), i + 20)

        for n in range(start, end + 1):
            print(f"{n}: {lines[n-1]}")

        print("\n" + "=" * 70 + "\n")


# In[13]:


from pathlib import Path

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

lines = app_path.read_text(encoding="utf-8").splitlines()

print("===== DETECTION LIST CREATION CODE =====\n")

for n in range(250, 290):
    if n <= len(lines):
        print(f"{n}: {lines[n-1]}")


# In[14]:


from pathlib import Path
import shutil
import py_compile
from datetime import datetime

# ============================================================
# APP PATH
# ============================================================

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

if not app_path.exists():
    raise FileNotFoundError(f"app.py not found:\n{app_path}")

print("✅ app.py found")


# ============================================================
# CREATE SAFETY BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = app_path.with_name(
    f"app_before_severity_{timestamp}.py"
)

shutil.copy2(app_path, backup_path)

print("✅ Safety backup created:")
print(backup_path)


# ============================================================
# READ APP.PY
# ============================================================

code = app_path.read_text(encoding="utf-8")


# ============================================================
# OLD DETECTION EXTRACTION BLOCK
# ============================================================

old_detection_block = '''        detection_list.append({
            "class": class_name,
            "confidence": confidence
        })'''


# ============================================================
# NEW DETECTION EXTRACTION BLOCK
# Adds bounding-box area ratio
# ============================================================

new_detection_block = '''        # Get bounding-box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        box_area = box_width * box_height

        # Original image dimensions
        image_height, image_width = result.orig_shape
        image_area = image_width * image_height

        # Percentage of image covered by this detection
        area_ratio = (
            (box_area / image_area) * 100
            if image_area > 0
            else 0.0
        )

        detection_list.append({
            "class": class_name,
            "confidence": confidence,
            "area_ratio": area_ratio
        })'''


# ============================================================
# REPLACE DETECTION BLOCK
# ============================================================

if '"area_ratio": area_ratio' not in code:

    if old_detection_block not in code:
        raise RuntimeError(
            "Expected detection_list block was not found. "
            "No changes written."
        )

    code = code.replace(
        old_detection_block,
        new_detection_block,
        1
    )

    print("✅ Bounding-box area calculation added")

else:
    print("ℹ️ Bounding-box area calculation already exists")


# ============================================================
# OLD SUMMARY BLOCK
# ============================================================

old_summary_block = '''            summary_lines.append(
                f"{class_name}: {count} detection(s) "
                f"(confidence scores: {confidence_text})"
            )'''


# ============================================================
# NEW SUMMARY BLOCK
#
# IMPORTANT:
# Severity is NOT based on confidence.
# Thresholds come from observed test-set box-area quartiles.
# ============================================================

new_summary_block = '''            # -----------------------------------------
            # Rule-based visual severity estimation
            # -----------------------------------------
            # This is NOT a trained structural severity
            # classifier. It is a visual extent estimate
            # based on bounding-box area relative to image
            # area, using thresholds derived from the
            # observed test-set detection distribution.

            class_lower = class_name.lower()

            if class_lower in ["crack", "hole"]:

                class_area_ratios = [
                    item["area_ratio"]
                    for item in detection_list
                    if item["class"] == class_name
                ]

                # Use the largest visible defect extent
                # for the class-level estimate
                max_area_ratio = max(class_area_ratios)

                if class_lower == "crack":

                    if max_area_ratio < 9.0139:
                        visual_severity = "Low"
                    elif max_area_ratio <= 28.4021:
                        visual_severity = "Medium"
                    else:
                        visual_severity = "High"

                else:  # hole

                    if max_area_ratio < 3.2815:
                        visual_severity = "Low"
                    elif max_area_ratio <= 14.1443:
                        visual_severity = "Medium"
                    else:
                        visual_severity = "High"

                summary_lines.append(
                    f"{class_name}: {count} detection(s) "
                    f"(confidence scores: {confidence_text}) | "
                    f"Visual severity estimate: {visual_severity} "
                    f"(largest box area: {max_area_ratio:.2f}% of image)"
                )

            else:

                # Normal is not assigned defect severity
                summary_lines.append(
                    f"{class_name}: {count} detection(s) "
                    f"(confidence scores: {confidence_text})"
                )'''


# ============================================================
# REPLACE SUMMARY BLOCK
# ============================================================

if "Visual severity estimate:" not in code:

    if old_summary_block not in code:
        raise RuntimeError(
            "Expected summary block was not found. "
            "No changes written."
        )

    code = code.replace(
        old_summary_block,
        new_summary_block,
        1
    )

    print("✅ Rule-based visual severity logic added")

else:
    print("ℹ️ Visual severity logic already exists")


# ============================================================
# SAVE UPDATED APP.PY
# ============================================================

app_path.write_text(
    code,
    encoding="utf-8"
)

print("✅ Updated app.py saved")


# ============================================================
# SYNTAX CHECK
# ============================================================

try:

    py_compile.compile(
        str(app_path),
        doraise=True
    )

    print("✅ Python syntax check passed")

except Exception as e:

    print("\n❌ SYNTAX CHECK FAILED")
    print(e)

    print("\nRestoring original app.py...")

    shutil.copy2(
        backup_path,
        app_path
    )

    print("✅ Original app.py restored")

    raise


# ============================================================
# FINAL VERIFICATION
# ============================================================

final_code = app_path.read_text(
    encoding="utf-8"
)

checks = {
    "Bounding-box coordinates":
        "box.xyxy[0].tolist()" in final_code,

    "Area ratio calculation":
        '"area_ratio": area_ratio' in final_code,

    "Crack thresholds":
        "9.0139" in final_code
        and "28.4021" in final_code,

    "Hole thresholds":
        "3.2815" in final_code
        and "14.1443" in final_code,

    "Visual severity output":
        "Visual severity estimate:" in final_code,

    "Normal excluded from severity":
        'if class_lower in ["crack", "hole"]' in final_code,
}


print("\n========================================")
print("SEVERITY FEATURE VERIFICATION")
print("========================================")

all_good = True

for name, result in checks.items():

    if result:
        print(f"✅ {name}: OK")
    else:
        print(f"❌ {name}: FAILED")
        all_good = False


if all_good:

    print("\n========================================")
    print("🎉 SUCCESS")
    print("========================================")

    print(
        "Rule-based visual severity estimation "
        "successfully added."
    )

    print("\nCrack thresholds:")
    print(" • Low: < 9.0139%")
    print(" • Medium: 9.0139% to 28.4021%")
    print(" • High: > 28.4021%")

    print("\nHole thresholds:")
    print(" • Low: < 3.2815%")
    print(" • Medium: 3.2815% to 14.1443%")
    print(" • High: > 14.1443%")

    print("\n⚠️ This is a rule-based VISUAL severity estimate.")
    print("It is NOT a trained structural severity classifier.")

    print("\nBackup:")
    print(backup_path)

else:

    print("\n❌ Verification failed.")
    print("Restoring original app.py for safety...")

    shutil.copy2(
        backup_path,
        app_path
    )

    print("✅ Original app.py restored.")


# In[15]:


from pathlib import Path
from datetime import datetime
import shutil
import py_compile
import re

# ============================================================
# CONFIG
# ============================================================

project_dir = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

app_path = project_dir / "app.py"

if not app_path.exists():
    raise FileNotFoundError(f"app.py not found:\n{app_path}")

print("✅ app.py found")

# ============================================================
# BACKUP
# ============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

backup_path = project_dir / f"app_before_clear_labels_{timestamp}.py"

shutil.copy2(app_path, backup_path)

print("✅ Safety backup created:")
print(backup_path)

# ============================================================
# READ CURRENT APP
# ============================================================

original_code = app_path.read_text(encoding="utf-8")
updated_code = original_code

try:

    # ========================================================
    # STEP 1 — ADD cv2 IMPORT IF NEEDED
    # ========================================================

    if not re.search(r"^\s*import\s+cv2\s*$", updated_code, re.MULTILINE):

        lines = updated_code.splitlines()

        insert_index = 0

        # Add near the other imports
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_index = i + 1

        lines.insert(insert_index, "import cv2")

        updated_code = "\n".join(lines) + "\n"

        print("✅ cv2 import added")

    else:
        print("✅ cv2 import already exists")

    # ========================================================
    # STEP 2 — FIND OLD result.plot() VISUALIZATION
    # ========================================================

    old_block = """    # Create annotated prediction image
    annotated_image = result.plot()

    # Convert BGR to RGB
    annotated_image = annotated_image[:, :, ::-1]"""

    if old_block not in updated_code:
        raise RuntimeError(
            "Could not find the expected result.plot() block.\n"
            "No changes were kept."
        )

    # ========================================================
    # STEP 3 — REPLACE WITH CUSTOM CLEAR ANNOTATION
    # ========================================================

    new_block = """    # =================================================
    # Create custom annotated prediction image
    # This avoids overlapping YOLO confidence labels
    # =================================================

    annotated_image = result.orig_img.copy()

    # Store occupied label regions so labels can be moved
    # when detections overlap.
    occupied_label_regions = []

    for detection_index, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Draw bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        label = f"{class_name} {confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        padding = 6

        # Preferred position is immediately above the box.
        label_x = max(0, x1)
        label_y = max(
            text_height + padding,
            y1 - 8
        )

        # Keep labels inside the image width.
        image_height, image_width = annotated_image.shape[:2]

        if label_x + text_width + (padding * 2) > image_width:
            label_x = max(
                0,
                image_width - text_width - (padding * 2)
            )

        # If a label overlaps a previous label,
        # move it downward until it becomes readable.
        max_attempts = 20

        for _ in range(max_attempts):

            region_left = label_x
            region_top = label_y - text_height - padding
            region_right = label_x + text_width + (padding * 2)
            region_bottom = label_y + baseline + padding

            overlaps = False

            for old_left, old_top, old_right, old_bottom in occupied_label_regions:

                if not (
                    region_right < old_left
                    or region_left > old_right
                    or region_bottom < old_top
                    or region_top > old_bottom
                ):
                    overlaps = True
                    break

            if not overlaps:
                break

            label_y += text_height + baseline + (padding * 2)

            # If moving down would leave the image,
            # place the label inside its bounding box.
            if label_y + baseline + padding >= image_height:
                label_y = min(
                    image_height - baseline - padding,
                    y1 + text_height + padding + 5
                )
                label_x = min(
                    max(0, x1 + 5),
                    max(0, image_width - text_width - (padding * 2))
                )
                break

        region_left = label_x
        region_top = max(0, label_y - text_height - padding)
        region_right = min(
            image_width - 1,
            label_x + text_width + (padding * 2)
        )
        region_bottom = min(
            image_height - 1,
            label_y + baseline + padding
        )

        occupied_label_regions.append(
            (
                region_left,
                region_top,
                region_right,
                region_bottom
            )
        )

        # Draw solid label background
        cv2.rectangle(
            annotated_image,
            (region_left, region_top),
            (region_right, region_bottom),
            (255, 255, 255),
            -1
        )

        # Draw dark text for maximum readability
        cv2.putText(
            annotated_image,
            label,
            (
                label_x + padding,
                label_y
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    # Convert BGR to RGB for Streamlit
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )"""

    updated_code = updated_code.replace(
        old_block,
        new_block,
        1
    )

    print("✅ Default YOLO annotation replaced")
    print("✅ Clear confidence labels added")
    print("✅ Label-overlap handling added")

    # ========================================================
    # WRITE UPDATED APP
    # ========================================================

    app_path.write_text(
        updated_code,
        encoding="utf-8"
    )

    print("✅ Updated app.py saved")

    # ========================================================
    # SYNTAX CHECK
    # ========================================================

    py_compile.compile(
        str(app_path),
        doraise=True
    )

    print("✅ Python syntax check passed")

    print("\n" + "=" * 55)
    print("🎉 CLEAR YOLO LABEL FEATURE SUCCESSFULLY ADDED")
    print("=" * 55)

    print(
        "\nThe annotated image will now display each detection "
        "with its own readable class + confidence label."
    )

    print("\nExample:")
    print(" • hole 0.68")
    print(" • hole 0.65")
    print(" • hole 0.59")

    print("\nExisting detection logic was NOT changed.")
    print("Existing confidence threshold remains unchanged.")
    print("Severity logic was NOT changed.")

    print("\nBackup:")
    print(backup_path)

except Exception as e:

    print("\n❌ UPDATE FAILED")
    print(e)

    print("\nRestoring original app.py from backup...")

    shutil.copy2(
        backup_path,
        app_path
    )

    print("✅ Original app.py restored.")


# In[16]:


from pathlib import Path

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

text = app_path.read_text(encoding="utf-8")

keywords = [
    "cv2.rectangle",
    "cv2.putText",
    "label_y",
    "used_label",
    "annotated_image"
]

lines = text.splitlines()

print("===== CURRENT ANNOTATION CODE =====\n")

found = False

for i, line in enumerate(lines):
    if any(keyword in line for keyword in keywords):
        found = True

        start = max(0, i - 12)
        end = min(len(lines), i + 20)

        for j in range(start, end):
            print(f"{j+1}: {lines[j]}")

        print("\n" + "=" * 70 + "\n")

if not found:
    print("❌ Annotation code not found.")


# In[17]:


from pathlib import Path
from datetime import datetime
import shutil
import py_compile

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

print("✅ app.py found")

# Backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = app_path.with_name(
    f"app_before_final_label_layout_{timestamp}.py"
)

shutil.copy2(app_path, backup_path)
print("✅ Safety backup created:")
print(backup_path)

text = app_path.read_text(encoding="utf-8")

start_marker = "    # Store occupied label regions so labels can be moved"
end_marker = "    # Convert BGR to RGB for Streamlit"

start = text.find(start_marker)
end = text.find(end_marker)

if start == -1 or end == -1:
    raise RuntimeError(
        "Could not locate the existing annotation block. "
        "app.py was NOT modified."
    )

new_block = '''    # =================================================
    # FINAL CLEAR ANNOTATION LAYOUT
    # Bounding boxes remain on detected objects.
    # Detection labels are stacked clearly at top-left.
    # =================================================

    label_items = []

    for detection_index, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Draw actual detection bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        label_items.append(
            f"{class_name} {confidence:.2f}"
        )

    # Draw readable stacked labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    padding = 6
    margin = 10
    current_y = margin

    for label in label_items:

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        box_left = margin
        box_top = current_y
        box_right = box_left + text_width + (padding * 2)
        box_bottom = box_top + text_height + baseline + (padding * 2)

        cv2.rectangle(
            annotated_image,
            (box_left, box_top),
            (box_right, box_bottom),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            annotated_image,
            label,
            (
                box_left + padding,
                box_top + padding + text_height
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        current_y = box_bottom + 5

'''

updated_text = text[:start] + new_block + text[end:]

app_path.write_text(updated_text, encoding="utf-8")

print("✅ Final clear label layout added")
print("✅ Detection logic unchanged")
print("✅ Confidence threshold unchanged")
print("✅ Severity logic unchanged")
print("✅ RAG/LLM logic unchanged")

try:
    py_compile.compile(str(app_path), doraise=True)

    print("\n========================================")
    print("🎉 FINAL UI FIX SUCCESS")
    print("========================================")
    print("✅ Python syntax check passed")
    print("✅ Labels will be displayed as a clean list")
    print("✅ Bounding boxes remain on detected objects")
    print("\nBackup:")
    print(backup_path)

except Exception as e:
    print("\n❌ SYNTAX CHECK FAILED")
    print(e)
    print("\nRestoring backup...")

    shutil.copy2(backup_path, app_path)

    print("✅ Original app.py restored")
    raise


# In[18]:


from pathlib import Path

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

text = app_path.read_text(encoding="utf-8")

keywords = [
    "clear",
    "label_positions",
    "cv2.rectangle",
    "cv2.putText",
    "detection_list.append"
]

lines = text.splitlines()

for i, line in enumerate(lines):
    if any(k.lower() in line.lower() for k in keywords):
        start = max(0, i - 12)
        end = min(len(lines), i + 25)

        print("\n" + "=" * 80)
        for j in range(start, end):
            print(f"{j+1}: {lines[j]}")


# In[19]:


from pathlib import Path
from datetime import datetime
import shutil
import py_compile

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

# -------------------------------------------------
# 1. Safety backup
# -------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = app_path.with_name(
    f"app_before_detection_ids_{timestamp}.py"
)

shutil.copy2(app_path, backup_path)

print("✅ app.py found")
print("✅ Safety backup created:")
print(backup_path)

text = app_path.read_text(encoding="utf-8")

# -------------------------------------------------
# 2. Locate current annotation section
# -------------------------------------------------
start_marker = "    label_items = []"
end_marker = "    # Convert BGR to RGB for Streamlit"

start = text.find(start_marker)
end = text.find(end_marker, start)

if start == -1 or end == -1:
    raise RuntimeError(
        "Could not locate the current annotation section. "
        "Original app.py has NOT been modified."
    )

# -------------------------------------------------
# 3. New annotation system
# -------------------------------------------------
new_block = '''    label_items = []

    # Separate counters for readable detection IDs
    class_id_counters = {}

    for detection_index, box in enumerate(result.boxes):

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Create readable prefix:
        # hole -> H1, H2...
        # crack -> C1, C2...
        # normal -> N1, N2...
        prefix = class_name[0].upper()

        class_id_counters[class_name] = (
            class_id_counters.get(class_name, 0) + 1
        )

        detection_id = (
            f"{prefix}{class_id_counters[class_name]}"
        )

        # Draw actual detection bounding box
        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3
        )

        # -------------------------------------------------
        # Put detection ID directly inside/near its box
        # -------------------------------------------------
        id_font = cv2.FONT_HERSHEY_SIMPLEX
        id_scale = 0.75
        id_thickness = 2

        (id_w, id_h), id_base = cv2.getTextSize(
            detection_id,
            id_font,
            id_scale,
            id_thickness
        )

        id_x = x1 + 5
        id_y = y1 + id_h + 10

        # Keep label inside image
        if id_y >= y2:
            id_y = max(id_h + 5, y1 - 5)

        cv2.rectangle(
            annotated_image,
            (id_x - 3, id_y - id_h - 5),
            (id_x + id_w + 5, id_y + id_base + 3),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            annotated_image,
            detection_id,
            (id_x, id_y),
            id_font,
            id_scale,
            (0, 0, 0),
            id_thickness,
            cv2.LINE_AA
        )

        # Store matching readable summary label
        label_items.append(
            f"{detection_id} | {class_name} | {confidence:.2f}"
        )

    # -------------------------------------------------
    # Draw readable stacked legend at top-left
    # -------------------------------------------------
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    padding = 6
    margin = 10
    current_y = margin

    for label in label_items:

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        box_left = margin
        box_top = current_y
        box_right = box_left + text_width + (padding * 2)
        box_bottom = (
            box_top
            + text_height
            + baseline
            + (padding * 2)
        )

        cv2.rectangle(
            annotated_image,
            (box_left, box_top),
            (box_right, box_bottom),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            annotated_image,
            label,
            (
                box_left + padding,
                box_top + padding + text_height
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        current_y = box_bottom + 5

'''

updated_text = (
    text[:start]
    + new_block
    + text[end:]
)

app_path.write_text(updated_text, encoding="utf-8")

print("✅ Detection IDs added")
print("✅ Bounding-box ID labels added")
print("✅ Top-left matching legend added")

# -------------------------------------------------
# 4. Syntax verification
# -------------------------------------------------
try:
    py_compile.compile(
        str(app_path),
        doraise=True
    )

    print("✅ Python syntax check passed")

except Exception:
    shutil.copy2(backup_path, app_path)
    print("❌ Syntax error detected")
    print("✅ Original app.py automatically restored")
    raise

print()
print("=" * 60)
print("🎉 DETECTION IDENTIFICATION FEATURE ADDED")
print("=" * 60)

print("""
Expected example:

H1 | hole | 0.68
H2 | hole | 0.65
H3 | hole | 0.59

Each bounding box will also contain:
H1
H2
H3

Therefore each confidence score can be matched
directly to its detected object.
""")

print("Backup:")
print(backup_path)


# In[20]:


from pathlib import Path
from datetime import datetime
import shutil
import py_compile

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

# Backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = app_path.with_name(
    f"app_before_id_position_fix_{timestamp}.py"
)

shutil.copy2(app_path, backup_path)

print("✅ Safety backup created:")
print(backup_path)

text = app_path.read_text(encoding="utf-8")

# Find the current block that positions the ID
old_block = '''        id_x = x1 + 5
        id_y = y1 + id_h + 10

        # Keep label inside image
        if id_y >= y2:
            id_y = max(id_h + 5, y1 - 5)
'''

new_block = '''        # Place ID near the top-right of its own bounding box.
        # This makes overlapping detections easier to distinguish.
        image_h, image_w = annotated_image.shape[:2]

        id_x = x2 + 8
        id_y = y1 + id_h + 5

        # If there is not enough room on the right,
        # place the ID just inside the right edge.
        if id_x + id_w + 8 >= image_w:
            id_x = max(5, x2 - id_w - 8)

        # Keep ID vertically inside the image.
        id_y = max(id_h + 8, min(id_y, image_h - 8))
'''

if old_block not in text:
    raise RuntimeError(
        "Current ID-position code was not found. "
        "app.py was NOT modified."
    )

text = text.replace(old_block, new_block, 1)

app_path.write_text(text, encoding="utf-8")

print("✅ Detection ID positioning improved")

# Syntax test
try:
    py_compile.compile(
        str(app_path),
        doraise=True
    )

    print("✅ Python syntax check passed")

except Exception:
    shutil.copy2(backup_path, app_path)

    print("❌ Syntax problem detected")
    print("✅ Previous app.py restored automatically")
    raise

print()
print("=" * 60)
print("🎉 BOUNDING-BOX ID POSITION FIX COMPLETE")
print("=" * 60)
print()
print("H1 / H2 / H3 will now be positioned")
print("near the top-right side of their own bounding boxes.")
print()
print("No YOLO prediction logic changed.")
print("No confidence values changed.")
print("No severity logic changed.")
print()
print("Backup:")
print(backup_path)


# In[ ]:


#evidance


# In[ ]:





# In[ ]:





# In[21]:


from pathlib import Path

test_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
    r"\Crack_Hole_Normal_Dataset\images\test"
)

files = list(test_path.rglob("*"))

print("Test folder:", test_path)
print("Total files:", len([f for f in files if f.is_file()]))

print("\nFirst 30 image files:")
for f in [x for x in files if x.is_file()][:30]:
    print(f)


# In[22]:


from pathlib import Path

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

test_images = project_root / "Crack_Hole_Normal_Dataset" / "images" / "test"
test_labels = project_root / "Crack_Hole_Normal_Dataset" / "labels" / "test"

print("Test images folder exists:", test_images.exists())
print("Test labels folder exists:", test_labels.exists())

image_files = list(test_images.glob("*"))
label_files = list(test_labels.glob("*.txt")) if test_labels.exists() else []

print("\nNumber of test images:", len(image_files))
print("Number of test label files:", len(label_files))

print("\nFirst 10 label files:")
for f in label_files[:10]:
    print(f.name)

if label_files:
    print("\nExample label content:")
    print("File:", label_files[0].name)
    print(label_files[0].read_text())


# In[23]:


from pathlib import Path
from collections import Counter

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

test_images = project_root / "Crack_Hole_Normal_Dataset" / "images" / "test"
test_labels = project_root / "Crack_Hole_Normal_Dataset" / "labels" / "test"

class_names = {
    0: "normal",
    1: "crack",
    2: "hole"
}

class_image_counts = Counter()
class_examples = {
    0: [],
    1: [],
    2: []
}

matched_labels = 0
missing_labels = []

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_files = [
    f for f in test_images.iterdir()
    if f.suffix.lower() in image_extensions
]

for image_path in image_files:

    label_path = test_labels / (image_path.stem + ".txt")

    if not label_path.exists():
        missing_labels.append(image_path.name)
        continue

    matched_labels += 1

    lines = label_path.read_text().strip().splitlines()

    # Get unique classes appearing in this image
    classes_in_image = set()

    for line in lines:
        parts = line.split()

        if parts:
            class_id = int(float(parts[0]))

            if class_id in class_names:
                classes_in_image.add(class_id)

    for class_id in classes_in_image:
        class_image_counts[class_id] += 1

        if len(class_examples[class_id]) < 10:
            class_examples[class_id].append(image_path.name)


print("===== TEST DATASET CLASS CHECK =====")

print("\nTotal test images:", len(image_files))
print("Matched image-label pairs:", matched_labels)
print("Images without labels:", len(missing_labels))

print("\n===== IMAGES CONTAINING EACH CLASS =====")

for class_id, class_name in class_names.items():
    print(
        f"{class_name.upper()}: "
        f"{class_image_counts[class_id]} images"
    )

print("\n===== EXAMPLE IMAGE FILES =====")

for class_id, class_name in class_names.items():

    print(f"\n{class_name.upper()} examples:")

    for filename in class_examples[class_id]:
        print(" ", filename)


# Find label files that do not have matching test images
image_stems = {f.stem for f in image_files}

extra_labels = [
    f.name
    for f in test_labels.glob("*.txt")
    if f.stem not in image_stems
]

print("\n===== EXTRA LABEL FILES =====")

if extra_labels:
    for filename in extra_labels:
        print(filename)
else:
    print("None")


# In[24]:


from pathlib import Path
from ultralytics import YOLO
import shutil

# =========================================================
# PATHS
# =========================================================

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

test_images = (
    project_root /
    "Crack_Hole_Normal_Dataset" /
    "images" /
    "test"
)

test_labels = (
    project_root /
    "Crack_Hole_Normal_Dataset" /
    "labels" /
    "test"
)

model_path = (
    project_root /
    "runs" /
    "detect" /
    "NHPT_Project_v2" /
    "Fine_Tuned-2" /
    "weights" /
    "best.pt"
)

output_root = project_root / "Report_Evidence" / "Example_Predictions"

# Start with a clean evidence folder
if output_root.exists():
    shutil.rmtree(output_root)

output_root.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOAD EXISTING TRAINED MODEL
# NO TRAINING IS PERFORMED
# =========================================================

model = YOLO(str(model_path))

print("Model loaded successfully.")
print("Classes:", model.names)

# =========================================================
# FIND SINGLE-CLASS EXAMPLE IMAGES
# =========================================================

class_names = {
    0: "normal",
    1: "crack",
    2: "hole"
}

selected = {
    0: [],
    1: [],
    2: []
}

image_extensions = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

image_files = sorted([
    f for f in test_images.iterdir()
    if f.suffix.lower() in image_extensions
])

for image_path in image_files:

    label_path = test_labels / (image_path.stem + ".txt")

    if not label_path.exists():
        continue

    lines = label_path.read_text().strip().splitlines()

    classes = []

    for line in lines:

        parts = line.split()

        if parts:
            classes.append(int(float(parts[0])))

    unique_classes = set(classes)

    # Select images containing only one class type
    if len(unique_classes) == 1:

        class_id = next(iter(unique_classes))

        if (
            class_id in selected
            and len(selected[class_id]) < 6
        ):
            selected[class_id].append(image_path)

    if all(len(v) >= 6 for v in selected.values()):
        break


print("\n===== SELECTED IMAGES =====")

for class_id, images in selected.items():

    print(
        f"\n{class_names[class_id].upper()} "
        f"({len(images)} images)"
    )

    for img in images:
        print(" ", img.name)

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

total_predictions = 0

for class_id, images in selected.items():

    class_name = class_names[class_id]

    class_output = output_root / class_name
    class_output.mkdir(parents=True, exist_ok=True)

    for number, image_path in enumerate(images, start=1):

        results = model.predict(
            source=str(image_path),
            conf=0.25,
            save=False,
            verbose=False
        )

        result = results[0]

        # Save annotated prediction image
        annotated = result.plot()

        output_file = (
            class_output /
            f"{class_name}_{number:02d}_prediction.jpg"
        )

        import cv2
        cv2.imwrite(str(output_file), annotated)

        # Print detected classes/confidences
        detections = []

        for box in result.boxes:

            detected_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append(
                f"{model.names[detected_id]} "
                f"{confidence:.2f}"
            )

        print(
            f"\n[{class_name.upper()} {number:02d}] "
            f"{image_path.name}"
        )

        if detections:
            print("Predictions:", ", ".join(detections))
        else:
            print("Predictions: No detection")

        print("Saved:", output_file)

        total_predictions += 1


print("\n" + "=" * 60)
print("REPORT EVIDENCE GENERATION COMPLETE")
print("=" * 60)

print("Total example images generated:", total_predictions)
print("Evidence folder:")
print(output_root)

print("\nNo model training was performed.")


# In[25]:


from pathlib import Path
import yaml

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

# ---------------------------------------------------------
# 1. DATA.YAML
# ---------------------------------------------------------

data_yaml = (
    project_root /
    "Crack_Hole_Normal_Dataset" /
    "data.yaml"
)

print("===== DATA.YAML =====")

with open(data_yaml, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print(data)

# ---------------------------------------------------------
# 2. FINAL TRAINING ARGS
# ---------------------------------------------------------

args_yaml = (
    project_root /
    "runs" /
    "detect" /
    "NHPT_Project_v2" /
    "Fine_Tuned-2" /
    "args.yaml"
)

print("\n===== FINE_TUNED-2 ARGS.YAML =====")

if args_yaml.exists():

    with open(args_yaml, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    print("data:", args.get("data"))
    print("model:", args.get("model"))
    print("epochs:", args.get("epochs"))
    print("classes:", args.get("classes"))

else:
    print("args.yaml not found")

# ---------------------------------------------------------
# 3. CHECK ACTUAL MODEL CLASS MAPPING
# ---------------------------------------------------------

from ultralytics import YOLO

model_path = (
    project_root /
    "runs" /
    "detect" /
    "NHPT_Project_v2" /
    "Fine_Tuned-2" /
    "weights" /
    "best.pt"
)

model = YOLO(str(model_path))

print("\n===== FINAL MODEL CLASS MAPPING =====")
print(model.names)

# ---------------------------------------------------------
# 4. CHECK SOME RAW LABELS
# ---------------------------------------------------------

test_labels = (
    project_root /
    "Crack_Hole_Normal_Dataset" /
    "labels" /
    "test"
)

examples = [
    "00016.txt",
    "00077.txt",
    "00101.txt",
    "5214136.txt",
    "5238356.txt"
]

print("\n===== RAW LABEL CHECK =====")

for name in examples:

    path = test_labels / name

    print(f"\n{name}")

    if path.exists():
        print(path.read_text(encoding="utf-8").strip())
    else:
        print("NOT FOUND")


# In[26]:


from pathlib import Path
from ultralytics import YOLO

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

model_path = (
    project_root /
    "runs" /
    "detect" /
    "NHPT_Project_v2" /
    "Fine_Tuned-2" /
    "weights" /
    "best.pt"
)

data_yaml = (
    project_root /
    "Crack_Hole_Normal_Dataset" /
    "data.yaml"
)

print("Loading final model...")
model = YOLO(str(model_path))

print("Model classes:", model.names)

print("\nRunning evaluation on TEST split...")
print("NO TRAINING is being performed.\n")

metrics = model.val(
    data=str(data_yaml),
    split="test",
    conf=0.001,
    iou=0.6,
    plots=True,
    verbose=False
)

print("\n========================================")
print("FINAL MODEL — TEST SET RESULTS")
print("========================================")

print(f"Precision:   {metrics.box.mp:.4f}")
print(f"Recall:      {metrics.box.mr:.4f}")
print(f"mAP50:       {metrics.box.map50:.4f}")
print(f"mAP50-95:    {metrics.box.map:.4f}")

precision = metrics.box.mp
recall = metrics.box.mr

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

print(f"F1 Score:    {f1:.4f}")

print("\n===== PER-CLASS RESULTS =====")

for i, class_name in model.names.items():

    p = metrics.box.p[i]
    r = metrics.box.r[i]
    ap50 = metrics.box.ap50[i]
    ap = metrics.box.ap[i]

    if p + r > 0:
        class_f1 = 2 * p * r / (p + r)
    else:
        class_f1 = 0

    print(f"\n{class_name.upper()}")
    print(f" Precision: {p:.4f}")
    print(f" Recall:    {r:.4f}")
    print(f" F1:        {class_f1:.4f}")
    print(f" mAP50:     {ap50:.4f}")
    print(f" mAP50-95:  {ap:.4f}")

print("\n========================================")
print("TEST EVALUATION COMPLETE")
print("========================================")
print("No model training was performed.")


# In[27]:


from pathlib import Path
from ultralytics import YOLO

root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

models = {
    "Original best.pt": (
        root / "NHPT_Project_v2" / "weights" / "best.pt"
    ),
    "Fine_Tuned-2 best.pt": (
        root / "runs" / "detect" / "NHPT_Project_v2" /
        "Fine_Tuned-2" / "weights" / "best.pt"
    )
}

data_yaml = (
    root / "Crack_Hole_Normal_Dataset" / "data.yaml"
)

print("=" * 65)
print("MODEL COMPARISON — TEST SET")
print("NO TRAINING WILL BE PERFORMED")
print("=" * 65)

for model_name, model_path in models.items():

    print(f"\n\n===== {model_name} =====")
    print("Path:", model_path)
    print("Exists:", model_path.exists())

    if not model_path.exists():
        continue

    model = YOLO(str(model_path))

    print("Classes:", model.names)

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False
    )

    p = metrics.box.mp
    r = metrics.box.mr

    f1 = (
        2 * p * r / (p + r)
        if (p + r) > 0
        else 0
    )

    print("\nOVERALL")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")

    print("\nPER CLASS")

    for i, class_name in model.names.items():

        cp = metrics.box.p[i]
        cr = metrics.box.r[i]

        cf1 = (
            2 * cp * cr / (cp + cr)
            if (cp + cr) > 0
            else 0
        )

        print(
            f"{class_name}: "
            f"P={cp:.4f}, "
            f"R={cr:.4f}, "
            f"F1={cf1:.4f}, "
            f"mAP50={metrics.box.ap50[i]:.4f}"
        )

print("\n" + "=" * 65)
print("COMPARISON COMPLETE")
print("=" * 65)


# In[28]:


from pathlib import Path
from collections import Counter

root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

dataset = root / "Crack_Hole_Normal_Dataset"

splits = ["train", "val", "test"]

print("=" * 65)
print("DATASET LABEL DISTRIBUTION CHECK")
print("=" * 65)

for split in splits:

    labels_dir = dataset / "labels" / split

    class_counts = Counter()
    total_boxes = 0
    total_files = 0

    for label_file in labels_dir.glob("*.txt"):

        # Ignore classes.txt or other metadata
        if label_file.name.lower() == "classes.txt":
            continue

        lines = label_file.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        if not lines:
            continue

        total_files += 1

        for line in lines:

            parts = line.split()

            if len(parts) >= 5:

                class_id = int(float(parts[0]))

                class_counts[class_id] += 1
                total_boxes += 1

    print(f"\n===== {split.upper()} =====")

    print("Label files:", total_files)
    print("Total bounding boxes:", total_boxes)

    for class_id in sorted(class_counts):

        print(
            f"Class {class_id}: "
            f"{class_counts[class_id]} boxes"
        )

print("\nExpected DATA.YAML mapping:")
print("Class 0 = normal")
print("Class 1 = crack")
print("Class 2 = hole")


# In[30]:


from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# PATHS
# =========================================================

root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

dataset = root / "Crack_Hole_Normal_Dataset"

class_names = {
    0: "NORMAL",
    1: "CRACK",
    2: "HOLE"
}

extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


# =========================================================
# FUNCTION TO FIND IMAGE
# =========================================================

def find_image(images_dir, stem):

    for ext in extensions:

        image_path = images_dir / f"{stem}{ext}"

        if image_path.exists():
            return image_path

    return None


# =========================================================
# FUNCTION TO GET 2 SINGLE-CLASS EXAMPLES
# =========================================================

def get_examples(split):

    images_dir = dataset / "images" / split
    labels_dir = dataset / "labels" / split

    examples = {
        0: [],
        1: [],
        2: []
    }

    for label_file in sorted(labels_dir.glob("*.txt")):

        if label_file.name.lower() == "classes.txt":
            continue

        text = label_file.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        class_ids = []

        for line in text.splitlines():

            parts = line.split()

            if len(parts) >= 5:
                class_ids.append(
                    int(float(parts[0]))
                )

        unique_classes = set(class_ids)

        # Select only images containing one class
        if len(unique_classes) == 1:

            class_id = list(unique_classes)[0]

            if (
                class_id in examples
                and len(examples[class_id]) < 2
            ):

                image_path = find_image(
                    images_dir,
                    label_file.stem
                )

                if image_path is not None:
                    examples[class_id].append(
                        image_path
                    )

        if all(
            len(examples[c]) >= 2
            for c in [0, 1, 2]
        ):
            break

    return examples


# =========================================================
# DISPLAY TRAIN / VAL / TEST
# =========================================================

for split in ["train", "val", "test"]:

    examples = get_examples(split)

    plt.figure(
        figsize=(15, 9),
        dpi=100
    )

    plot_number = 1

    for class_id in [0, 1, 2]:

        for image_path in examples[class_id]:

            plt.subplot(
                2,
                3,
                plot_number
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            plt.imshow(image)

            plt.title(
                f"Class {class_id} = "
                f"{class_names[class_id]}\n"
                f"{image_path.name}",
                fontsize=12
            )

            plt.axis("off")

            plot_number += 1

    plt.suptitle(
        f"{split.upper()} DATASET - CLASS CHECK",
        fontsize=18
    )

    plt.tight_layout()

    # Save image
    output_file = root / f"{split}_class_check.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    # IMPORTANT - display graph
    plt.show()

    print(
        f"✅ {split.upper()} graph saved:"
    )
    print(output_file)

print("\n===================================")
print("ALL CLASS CHECK GRAPHS COMPLETED")
print("===================================")


# In[ ]:





# In[1]:


from pathlib import Path

# CHANGE this only if your knowledge-base folder has a different location
project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

print("===== SEARCHING FOR KNOWLEDGE BASE FILES =====\n")

# Common document formats
extensions = {".pdf", ".txt", ".docx", ".md"}

documents = []

for file in project_root.rglob("*"):
    if file.is_file() and file.suffix.lower() in extensions:
        # Ignore obvious coursework/report/evidence files
        path_lower = str(file).lower()

        if not any(x in path_lower for x in [
            "report_evidence",
            "runs",
            ".ipynb_checkpoints"
        ]):
            documents.append(file)

print(f"Potential knowledge documents found: {len(documents)}\n")

for i, file in enumerate(documents, 1):
    print(f"{i}. {file.name}")
    print(f"   Folder: {file.parent}")
    print()


# In[2]:


from pathlib import Path

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

extensions = {".pdf", ".docx", ".md", ".txt"}

# Folders/files that are NOT RAG knowledge documents
exclude_terms = [
    ".venv",
    "site-packages",
    "crack_hole_normal_dataset",
    "labels",
    "images",
    "runs",
    "report_evidence",
    ".ipynb_checkpoints",
    "__pycache__"
]

documents = []

for file in project_root.rglob("*"):
    if not file.is_file():
        continue

    if file.suffix.lower() not in extensions:
        continue

    path_lower = str(file).lower()

    if any(term in path_lower for term in exclude_terms):
        continue

    documents.append(file)

print("=" * 70)
print("POTENTIAL RAG / KNOWLEDGE-BASE DOCUMENTS")
print("=" * 70)

print(f"\nTotal candidate documents: {len(documents)}\n")

for i, file in enumerate(documents, 1):
    print(f"{i}. {file.name}")
    print(f"   Folder: {file.parent}")
    print()

print("=" * 70)


# In[3]:


from pathlib import Path

project_root = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project"
)

keywords = [
    "FAISS",
    "vectorstore",
    "vector_store",
    "HuggingFaceEmbeddings",
    "OllamaEmbeddings",
    "RecursiveCharacterTextSplitter",
    "TextLoader",
    "PyPDFLoader",
    "DirectoryLoader",
    "from_documents",
    "similarity_search",
    "as_retriever"
]

exclude = [".venv", "site-packages", "__pycache__"]

print("=" * 70)
print("SEARCHING PROJECT CODE FOR RAG IMPLEMENTATION")
print("=" * 70)

found = []

for file in project_root.rglob("*"):
    if not file.is_file():
        continue

    if file.suffix.lower() not in {".py", ".ipynb"}:
        continue

    path_lower = str(file).lower()

    if any(x in path_lower for x in exclude):
        continue

    try:
        text = file.read_text(encoding="utf-8", errors="ignore")

        matches = [k for k in keywords if k.lower() in text.lower()]

        if matches:
            found.append((file, matches))

    except Exception:
        pass

print(f"\nFiles containing RAG-related code: {len(found)}\n")

for i, (file, matches) in enumerate(found, 1):
    print(f"{i}. {file.name}")
    print(f"   Path: {file}")
    print(f"   Found: {', '.join(matches)}")
    print()

print("=" * 70)


# In[4]:


from pathlib import Path

app_path = Path(
    r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\app.py"
)

text = app_path.read_text(encoding="utf-8", errors="ignore")
lines = text.splitlines()

keywords = [
    "FAISS",
    "load_local",
    "save_local",
    "similarity_search",
    "retriever",
    "embedding",
    "knowledge",
    "vector",
    "document"
]

print("=" * 70)
print("RAG-RELATED SECTIONS FROM FINAL app.py")
print("=" * 70)

printed = set()

for i, line in enumerate(lines):
    if any(k.lower() in line.lower() for k in keywords):
        
        start = max(0, i - 4)
        end = min(len(lines), i + 8)
        
        # avoid printing same block repeatedly
        block_id = (start, end)
        if block_id in printed:
            continue
        printed.add(block_id)

        print(f"\n--- Around line {i+1} ---")
        
        for j in range(start, end):
            print(f"{j+1:04d}: {lines[j]}")

print("\n" + "=" * 70)


# In[ ]:





# In[ ]:





# In[ ]:




