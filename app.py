
import streamlit as st
from ultralytics import YOLO
import os
from collections import Counter
import ollama
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import cv2

st.set_page_config(
    page_title="AI Surface Defect Inspection System",
    page_icon="🔍",
    layout="wide"
)

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



# Path to the trained YOLO model
model_path = r"C:\Users\HP\OneDrive\Desktop\ML Applications practical\ML CW NHPT_Project\runs\detect\NHPT_Project_v2\Fine_Tuned-2\weights\best.pt"

@st.cache_resource
def load_yolo_model():
    return YOLO(model_path)

# Load trained model
model = load_yolo_model()



# =================================================
# RAG KNOWLEDGE BASE
# =================================================

knowledge_source_titles = [
    "Crack Detection and Inspection Guide",
    "Hole Detection and Assessment Guide",
    "Normal Surface and Maintenance Guide",
    "Structural Inspection Recommendations",
    "Defect Confidence Score Interpretation Guide",
    "Defect Monitoring and Follow-up Guide"
]

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
]


# =================================================
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

    # =================================================
    # Create custom annotated prediction image
    # This avoids overlapping YOLO confidence labels
    # =================================================

    annotated_image = result.orig_img.copy()

    # =================================================
    # FINAL CLEAR ANNOTATION LAYOUT
    # Bounding boxes remain on detected objects.
    # Detection labels are stacked clearly at top-left.
    # =================================================

    label_items = []

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

        # Place ID near the top-right of its own bounding box.
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

    # Convert BGR to RGB for Streamlit
    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_BGR2RGB
    )

    # Display YOLO detection result
    st.subheader("YOLO Defect Detection Result")

    st.image(
        annotated_image,
        caption="Detected Surface Defects"
    )

    # ---------------------------------------------
    # Extract Exact Detection Results
    # ---------------------------------------------
    detection_list = []

    for box in result.boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        # Get bounding-box coordinates
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

            # -----------------------------------------
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
                )

        detection_summary = "\n".join(summary_lines)

    else:
        detection_summary = "No crack or hole defects were detected."

    # Display exact detection summary
    # Save latest YOLO result in session memory
    st.session_state.latest_detection_summary = detection_summary

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

    st.success("YOLO analysis completed successfully!")

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
    retrieved_context = "\n\n".join(
        retrieved_documents
    )


    # Save latest RAG context in session memory
    st.session_state.latest_retrieved_context = retrieved_context

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
            model=selected_llm_model,
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





# =================================================


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
    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )


    # Store retrieved knowledge with source labels
    retrieved_chat_documents = []

    for rank, doc_index in enumerate(indices[0]):

        if doc_index >= 0:

            # Retrieve the corresponding LangChain Document
            retrieved_document = langchain_documents[doc_index]

            retrieved_chat_documents.append({
                "source": retrieved_document.metadata["source"],
                "content": retrieved_document.page_content
            })


    # Combine retrieved document content as context
    retrieved_chat_context = "\n\n".join(
        document["content"]
        for document in retrieved_chat_documents
    )


    return (
        retrieved_chat_context,
        retrieved_chat_documents
    )


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

    for message in st.session_state.messages[:-1]:

        conversation_history += (
            f'{message["role"].upper()}: '
            f'{message["content"]}\n'
        )


    # ---------------------------------------------
    # Build grounded chatbot prompt
    # ---------------------------------------------

    # Build the grounded chatbot prompt using LangChain PromptTemplate
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
    )


    # ---------------------------------------------
    # Generate chatbot response using Ollama
    # ---------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Generating response..."
        ):

            chat_response = ollama.chat(
                model=selected_llm_model,
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

            # Display RAG sources immediately below current answer
            if fresh_chat_documents:

                with st.expander(
                    "📚 Sources used for this answer"
                ):

                    for document in fresh_chat_documents:

                        st.write(
                            f"- {document['source']}"
                        )


    # ---------------------------------------------
    # Save assistant response to memory
    # ---------------------------------------------

    # Prepare source labels used for this answer
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
    )

