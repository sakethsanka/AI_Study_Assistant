import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
import sqlite3
from datetime import datetime


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "study_assistant.db"


def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_name TEXT,
            total_questions INTEGER,
            correct_answers INTEGER,
            accuracy REAL,
            completed_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_quiz_result(
    quiz_name,
    total_questions,
    correct_answers
):
    if total_questions == 0:
        accuracy = 0
    else:
        accuracy = (
            correct_answers / total_questions
        ) * 100

    completed_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_history
        (
            quiz_name,
            total_questions,
            correct_answers,
            accuracy,
            completed_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        quiz_name,
        total_questions,
        correct_answers,
        accuracy,
        completed_at
    ))

    conn.commit()
    conn.close()


def get_quiz_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            quiz_name,
            total_questions,
            correct_answers,
            accuracy,
            completed_at
        FROM quiz_history
        ORDER BY id DESC
    """)

    history = cursor.fetchall()

    conn.close()

    return history


def clear_database_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM quiz_history")

    conn.commit()
    conn.close()


# Initialize database
init_database()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "❌ GROQ_API_KEY not found in your .env file."
    )
    st.stop()


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)

MODEL_NAME = "openai/gpt-oss-120b"


# ============================================================
# SESSION STATE
# ============================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []

if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = "No PDF uploaded"

if "current_pdf_pages" not in st.session_state:
    st.session_state.current_pdf_pages = 0

if "rag_chunk_count" not in st.session_state:
    st.session_state.rag_chunk_count = 0

if "last_learning_activity" not in st.session_state:
    st.session_state.last_learning_activity = "No activity yet"

if "last_ai_response" not in st.session_state:
    st.session_state.last_ai_response = ""


# ============================================================
# CHAT MEMORY
# ============================================================

def add_to_chat_history(user_message, ai_message):

    st.session_state.chat_history.append({
        "user": user_message,
        "assistant": ai_message
    })


def clear_chat_history():

    st.session_state.chat_history = []


# ============================================================
# AI FUNCTION
# ============================================================

def ask_ai(prompt):

    try:

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:

        return (
            "⚠️ **Groq could not generate a response.**\n\n"
            f"Error: {str(e)}"
        )


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    pdf_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            pdf_text += page_text + "\n"

    return pdf_text, len(reader.pages)


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=150
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(
    chunks,
    model
):

    if not chunks:
        return None

    embeddings = model.encode(
        list(chunks)
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    chunks,
    index,
    model,
    k=4
):

    if not chunks or index is None:
        return [], []

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    k = min(k, len(chunks))

    distances, indices = index.search(
        question_embedding,
        k
    )

    relevant_chunks = []
    valid_distances = []

    for position, index_number in enumerate(
        indices[0]
    ):

        if index_number >= 0:

            relevant_chunks.append(
                chunks[index_number]
            )

            valid_distances.append(
                distances[0][position]
            )

    return (
        relevant_chunks,
        valid_distances
    )


# ============================================================
# QUIZ PARSER
# ============================================================

def parse_quiz_response(text):

    questions = []

    pattern = re.compile(
        r"Question\s*(\d+)\s*:\s*(.*?)"
        r"\s*A\.\s*(.*?)"
        r"\s*B\.\s*(.*?)"
        r"\s*C\.\s*(.*?)"
        r"\s*D\.\s*(.*?)"
        r"\s*Correct\s*Answer\s*:\s*([ABCD])",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(text)

    for match in matches:

        question_text = match[1].strip()
        option_a = match[2].strip()
        option_b = match[3].strip()
        option_c = match[4].strip()
        option_d = match[5].strip()
        correct_answer = match[6].upper().strip()

        questions.append({
            "question": question_text,
            "options": {
                "A": option_a,
                "B": option_b,
                "C": option_c,
                "D": option_d
            },
            "answer": correct_answer
        })

    return questions


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(material):

    prompt = f"""
You are an AI college study assistant.

Create EXACTLY 5 multiple-choice questions
STRICTLY from the study material below.

Do not use unrelated general knowledge.

STUDY MATERIAL:

{material}

IMPORTANT:

Return ONLY this format:

Question 1:

[question]

A. [option]

B. [option]

C. [option]

D. [option]

Correct Answer: A

Question 2:

[question]

A. [option]

B. [option]

C. [option]

D. [option]

Correct Answer: B

Question 3:

[question]

A. [option]

B. [option]

C. [option]

D. [option]

Correct Answer: C

Question 4:

[question]

A. [option]

B. [option]

C. [option]

D. [option]

Correct Answer: D

Question 5:

[question]

A. [option]

B. [option]

C. [option]

D. [option]

Correct Answer: A

Rules:

- Exactly 5 questions.
- Exactly 4 options per question.
- Correct answer must be A, B, C or D.
- Every question must come from the material.
- Use simple English.
- Avoid duplicate questions.
"""

    response = ask_ai(prompt)

    parsed_questions = parse_quiz_response(
        response
    )

    return parsed_questions, response


# ============================================================
# RESET QUIZ
# ============================================================

def reset_quiz():

    st.session_state.quiz_data = []

    st.session_state.quiz_started = False

    st.session_state.quiz_submitted = False

    st.session_state.quiz_score = 0

    st.session_state.quiz_answers = {}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background-color: #0d1117;
}

section[data-testid="stSidebar"] {
    background-color: #171b27;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* HERO */

.hero {
    background:
        linear-gradient(
            135deg,
            #667eea,
            #764ba2
        );

    padding: 35px;

    border-radius: 20px;

    text-align: center;

    margin-bottom: 30px;
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 12px;
}

.hero-subtitle {
    color: white;
    font-size: 18px;
}


/* WHITE CARD */

.study-card {
    background-color: #ffffff;

    padding: 30px;

    border-radius: 18px;

    margin-bottom: 20px;

    box-shadow:
        0px 4px 15px
        rgba(0, 0, 0, 0.20);
}

.study-card * {
    color: #172033 !important;
}

.study-title {
    color: #172033 !important;

    font-size: 28px;

    font-weight: 700;

    margin-bottom: 10px;
}

.study-description {
    color: #4b5563 !important;

    font-size: 17px;
}


/* INPUT */

textarea {
    color: #172033 !important;

    background-color: #ffffff !important;
}

textarea::placeholder {
    color: #718096 !important;

    opacity: 1 !important;
}

input {
    color: #172033 !important;

    background-color: #ffffff !important;
}


/* BUTTON */

.stButton > button {

    background:
        linear-gradient(
            135deg,
            #667eea,
            #764ba2
        );

    color: white !important;

    border: none;

    border-radius: 10px;

    min-height: 50px;

    font-size: 16px;

    font-weight: 600;
}

.stButton > button:hover {
    color: white !important;
}


/* DOWNLOAD */

.stDownloadButton > button {

    background:
        linear-gradient(
            135deg,
            #667eea,
            #764ba2
        );

    color: white !important;

    border: none;

    border-radius: 10px;

    min-height: 45px;

    font-weight: 600;
}


/* RESULT BOX */

.result-box {

    background-color: #ffffff;

    padding: 25px;

    border-radius: 15px;

    border-left:
        5px solid #667eea;

    box-shadow:
        0px 4px 15px
        rgba(0, 0, 0, 0.15);

    margin-top: 20px;
}

.result-box,
.result-box * {
    color: #172033 !important;
}


/* SOURCE BOX */

.source-box {

    background-color: #f3f4f6;

    padding: 15px;

    border-radius: 10px;

    border-left:
        4px solid #764ba2;

    margin-bottom: 10px;
}

.source-box,
.source-box * {
    color: #172033 !important;
}


/* CHAT */

.chat-user {

    background-color: #667eea;

    color: white !important;

    padding: 15px;

    border-radius: 12px;

    margin-top: 15px;

    margin-bottom: 10px;
}

.chat-user * {
    color: white !important;
}

.chat-ai {

    background-color: #ffffff;

    color: #172033 !important;

    padding: 15px;

    border-radius: 12px;

    border-left: 4px solid #764ba2;

    margin-bottom: 15px;
}

.chat-ai * {
    color: #172033 !important;
}


/* QUIZ */

.quiz-card {

    background-color: #ffffff;

    padding: 25px;

    border-radius: 15px;

    margin-bottom: 20px;

    border-left: 5px solid #667eea;
}

.quiz-card,
.quiz-card * {
    color: #172033 !important;
}

.quiz-question {

    color: #172033 !important;

    font-size: 20px;

    font-weight: 700;

    margin-bottom: 15px;
}


/* SCORE */

.score-box {

    background:
        linear-gradient(
            135deg,
            #667eea,
            #764ba2
        );

    color: white !important;

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    margin: 25px 0;
}

.score-box * {
    color: white !important;
}

.score-number {

    font-size: 48px;

    font-weight: 800;
}


/* CORRECT */

.correct-box {

    background-color: #dcfce7;

    color: #166534 !important;

    padding: 15px;

    border-radius: 10px;

    margin-bottom: 15px;
}

.correct-box * {
    color: #166534 !important;
}


/* WRONG */

.wrong-box {

    background-color: #fee2e2;

    color: #991b1b !important;

    padding: 15px;

    border-radius: 10px;

    margin-bottom: 15px;
}

.wrong-box * {
    color: #991b1b !important;
}


/* INFO */

.info-box {

    background-color: #eef2ff;

    color: #172033 !important;

    padding: 15px;

    border-radius: 10px;

    margin-bottom: 15px;
}

.info-box * {
    color: #172033 !important;
}


/* DASHBOARD */

.dashboard-card {

    background-color: #ffffff;

    padding: 25px;

    border-radius: 18px;

    margin-bottom: 20px;

    border-left: 5px solid #667eea;

    box-shadow:
        0px 4px 15px
        rgba(0, 0, 0, 0.20);
}

.dashboard-card,
.dashboard-card * {
    color: #172033 !important;
}

.dashboard-number {

    font-size: 36px;

    font-weight: 800;

    color: #667eea !important;
}

.dashboard-label {

    font-size: 16px;

    font-weight: 600;

    color: #4b5563 !important;
}


/* HISTORY */

.history-card {

    background-color: #ffffff;

    padding: 25px;

    border-radius: 18px;

    margin-bottom: 20px;

    border-left: 5px solid #667eea;

    box-shadow:
        0px 4px 15px
        rgba(0, 0, 0, 0.20);
}

.history-card,
.history-card * {
    color: #172033 !important;
}

.history-title {

    font-size: 20px;

    font-weight: 700;

    color: #172033 !important;
}

.history-detail {

    font-size: 17px;

    color: #374151 !important;

    margin-top: 10px;
}


/* FOOTER */

.footer {

    text-align: center;

    color: #9ca3af;

    padding: 30px;

    margin-top: 40px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎓 AI Study Assistant"
)

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📚 Main Menu"
)

page = st.sidebar.radio(
    "Choose a page:",
    [
        "📖 Study Assistant",
        "📝 Interactive Quiz",
        "📚 Quiz History",
        "📊 Study Dashboard"
    ]
)


# ============================================================
# STUDY ASSISTANT SETTINGS
# ============================================================

if page == "📖 Study Assistant":

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "📚 Input Mode"
    )

    input_mode = st.sidebar.radio(
        "Choose how you want to provide content:",
        [
            "✍️ Text Input",
            "📄 PDF Upload"
        ]
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🛠️ Study Tool"
    )

    if input_mode == "📄 PDF Upload":

        tool = st.sidebar.radio(
            "Choose an AI tool:",
            [
                "📝 Summarize",
                "💡 Explain",
                "❓ Generate Quiz",
                "🔎 Ask Questions (RAG)"
            ]
        )

    else:

        tool = st.sidebar.radio(
            "Choose an AI tool:",
            [
                "📝 Summarize",
                "💡 Explain",
                "❓ Generate Quiz"
            ]
        )

    st.sidebar.markdown("---")

    if st.sidebar.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        clear_chat_history()

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

<div class="hero-title">
🎓 AI Study Assistant
</div>

<div class="hero-subtitle">
Study smarter with AI-powered summaries,
explanations, quizzes and PDF Q&A.
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1
# STUDY ASSISTANT
# ============================================================

if page == "📖 Study Assistant":

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.markdown(
            "## 💬 Conversation"
        )

        for chat in st.session_state.chat_history:

            st.markdown(
                f"""
<div class="chat-user">

<b>👤 You</b>

<br><br>

{chat["user"]}

</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
<div class="chat-ai">

<b>🤖 AI Assistant</b>

<br><br>

{chat["assistant"]}

</div>
""",
                unsafe_allow_html=True
            )


    # ========================================================
    # TEXT INPUT
    # ========================================================

    if input_mode == "✍️ Text Input":

        st.markdown(
            """
<div class="study-card">

<div class="study-title">
✏️ Enter Your Study Content
</div>

<div class="study-description">
Paste your notes, topic, question or study material below.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        with st.form(
            "text_ai_form"
        ):

            text = st.text_area(
                "Enter your content:",
                height=180,
                placeholder=
                "Paste your study material or type a topic/question..."
            )

            ai_assist = st.form_submit_button(
                "🤖 AI Assist",
                use_container_width=True
            )


        if ai_assist:

            if not text.strip():

                st.warning(
                    "⚠️ Please enter some content."
                )

            else:

                # ==================================================
                # SUMMARY
                # ==================================================

                if tool == "📝 Summarize":

                    prompt = f"""
You are an AI study assistant.

Use ONLY the study material below.

STUDY MATERIAL:

{text}

Create a clear summary for a college student.

Requirements:

- Use simple English.
- Use headings.
- Use bullet points.
- Keep important information.
- Keep technical terms.
- Do not add unrelated information.
- Make it useful for exam revision.
"""


                # ==================================================
                # EXPLAIN
                # ==================================================

                elif tool == "💡 Explain":

                    prompt = f"""
You are an AI study assistant.

Use ONLY the study material below.

STUDY MATERIAL:

{text}

Explain it to a beginner college student.

Requirements:

- Use very simple English.
- Explain step by step.
- Explain important terms.
- Give simple examples when useful.
- Highlight important points.
- Do not add unrelated information.
"""


                # ==================================================
                # QUIZ
                # ==================================================

                else:

                    with st.spinner(
                        "🤖 Creating your quiz..."
                    ):

                        quiz_data, raw_quiz = generate_quiz(
                            text
                        )

                    if len(quiz_data) != 5:

                        st.error(
                            "⚠️ The AI did not create exactly "
                            "5 questions. Please try again."
                        )

                        st.code(raw_quiz)

                    else:

                        st.session_state.quiz_data = quiz_data

                        st.session_state.quiz_started = True

                        st.session_state.quiz_submitted = False

                        st.session_state.quiz_score = 0

                        st.session_state.quiz_answers = {}

                        st.session_state.last_learning_activity = (
                            "Generated a quiz from text"
                        )

                        add_to_chat_history(
                            "Generated a quiz",
                            raw_quiz
                        )

                        st.success(
                            "✅ Quiz generated successfully!"
                        )

                        st.info(
                            "Go to 📝 Interactive Quiz "
                            "from the sidebar."
                        )

                    st.stop()


                # ==================================================
                # AI RESPONSE
                # ==================================================

                with st.spinner(
                    "🤖 Groq is generating your answer..."
                ):

                    answer = ask_ai(
                        prompt
                    )

                add_to_chat_history(
                    text,
                    answer
                )

                st.session_state.last_ai_response = answer

                if tool == "📝 Summarize":

                    st.session_state.last_learning_activity = (
                        "Generated a study summary"
                    )

                else:

                    st.session_state.last_learning_activity = (
                        "Generated a study explanation"
                    )

                st.markdown(
                    '<div class="result-box">',
                    unsafe_allow_html=True
                )

                st.subheader(
                    "🤖 AI Response"
                )

                st.markdown(
                    answer
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

                st.download_button(
                    "📥 Download AI Response",
                    data=answer,
                    file_name="ai_study_response.txt",
                    mime="text/plain",
                    use_container_width=True
                )


    # ========================================================
    # PDF MODE
    # ========================================================

    else:

        st.markdown(
            """
<div class="study-card">

<div class="study-title">
📄 Upload Your Study PDF
</div>

<div class="study-description">
Upload a PDF and summarize, explain,
create a quiz, or ask questions using
document-based retrieval.
</div>

</div>
""",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF file:",
            type=["pdf"]
        )


        if uploaded_file is not None:

            try:

                pdf_text, page_count = extract_pdf_text(
                    uploaded_file
                )

                if not pdf_text.strip():

                    st.warning(
                        "⚠️ Could not extract text from this PDF."
                    )

                else:

                    st.session_state.current_pdf_name = (
                        uploaded_file.name
                    )

                    st.session_state.current_pdf_pages = (
                        page_count
                    )

                    st.success(
                        f"✅ PDF uploaded successfully! "
                        f"Pages: {page_count}"
                    )

                    MAX_CHARS = 50000

                    if len(pdf_text) > MAX_CHARS:

                        pdf_text = pdf_text[:MAX_CHARS]

                        st.info(
                            "ℹ️ Only the first "
                            "50,000 characters are processed."
                        )

                    with st.expander(
                        "👀 Preview Extracted Text"
                    ):

                        st.text(
                            pdf_text[:3000]
                        )


                    # ==================================================
                    # PDF RAG
                    # ==================================================

                    if tool == "🔎 Ask Questions (RAG)":

                        st.markdown(
                            "### 🔎 Ask a Question About Your PDF"
                        )

                        with st.form(
                            "pdf_question_form"
                        ):

                            question = st.text_input(
                                "Enter your question:",
                                placeholder=
                                "Type your question about the PDF..."
                            )

                            ai_assist = st.form_submit_button(
                                "🤖 AI Assist",
                                use_container_width=True
                            )


                        if ai_assist:

                            if not question.strip():

                                st.warning(
                                    "⚠️ Please enter a question."
                                )

                            else:

                                try:

                                    with st.spinner(
                                        "🧠 Loading embedding model..."
                                    ):

                                        model = load_embedding_model()


                                    with st.spinner(
                                        "✂️ Creating document chunks..."
                                    ):

                                        chunks = create_chunks(
                                            pdf_text
                                        )

                                    st.session_state.pdf_chunks = chunks

                                    st.session_state.rag_chunk_count = len(
                                        chunks
                                    )

                                    st.info(
                                        f"📚 Created {len(chunks)} "
                                        f"document chunks."
                                    )


                                    with st.spinner(
                                        "🔎 Preparing document search..."
                                    ):

                                        index = build_faiss_index(
                                            tuple(chunks),
                                            model
                                        )

                                    st.session_state.pdf_index = index


                                    with st.spinner(
                                        "🔎 Searching relevant information..."
                                    ):

                                        (
                                            relevant_chunks,
                                            distances
                                        ) = retrieve_relevant_chunks(
                                            question,
                                            chunks,
                                            index,
                                            model,
                                            k=4
                                        )


                                    context = (
                                        "\n\n--- SOURCE CHUNK ---\n\n"
                                        .join(
                                            relevant_chunks
                                        )
                                    )


                                    prompt = f"""
You are a helpful AI study assistant.

Answer the question using ONLY the
retrieved PDF context.

RETRIEVED PDF CONTEXT:

{context}

QUESTION:

{question}

Rules:

- Do not invent information.
- Do not use unrelated general knowledge.
- Use simple English.
- If the answer is not found, say:

"I could not find the answer in the uploaded document."
"""


                                    with st.spinner(
                                        "🤖 Groq is preparing your answer..."
                                    ):

                                        answer = ask_ai(
                                            prompt
                                        )


                                    add_to_chat_history(
                                        question,
                                        answer
                                    )

                                    st.session_state.last_ai_response = answer

                                    st.session_state.last_learning_activity = (
                                        "Asked a question using PDF RAG"
                                    )


                                    st.markdown(
                                        '<div class="result-box">',
                                        unsafe_allow_html=True
                                    )

                                    st.subheader(
                                        "🤖 AI Answer"
                                    )

                                    st.markdown(
                                        answer
                                    )

                                    st.markdown(
                                        '</div>',
                                        unsafe_allow_html=True
                                    )


                                    st.markdown(
                                        "### 📚 Retrieved Source Context"
                                    )


                                    for i, chunk in enumerate(
                                        relevant_chunks
                                    ):

                                        st.markdown(
                                            f"""
<div class="source-box">

<b>🔹 Source {i + 1}</b>

<br><br>

{chunk}

<br><br>

<b>📊 Retrieval distance:</b>

{distances[i]:.4f}

</div>
""",
                                            unsafe_allow_html=True
                                        )


                                    st.download_button(
                                        "📥 Download AI Answer",
                                        data=answer,
                                        file_name=
                                        "pdf_question_answer.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )


                                except Exception as e:

                                    st.error(
                                        "⚠️ Unable to process "
                                        "the PDF question."
                                    )

                                    with st.expander(
                                        "Technical details"
                                    ):

                                        st.code(
                                            str(e)
                                        )


                    # ==================================================
                    # NORMAL PDF TOOLS
                    # ==================================================

                    else:

                        if st.button(
                            "🚀 Generate AI Response",
                            use_container_width=True
                        ):


                            # ==========================================
                            # PDF QUIZ
                            # ==========================================

                            if tool == "❓ Generate Quiz":

                                with st.spinner(
                                    "🤖 Creating quiz from PDF..."
                                ):

                                    quiz_data, raw_quiz = generate_quiz(
                                        pdf_text
                                    )


                                if len(quiz_data) != 5:

                                    st.error(
                                        "⚠️ Could not create "
                                        "exactly 5 questions."
                                    )

                                    st.code(
                                        raw_quiz
                                    )

                                else:

                                    st.session_state.quiz_data = quiz_data

                                    st.session_state.quiz_started = True

                                    st.session_state.quiz_submitted = False

                                    st.session_state.quiz_score = 0

                                    st.session_state.quiz_answers = {}

                                    st.session_state.last_learning_activity = (
                                        "Generated a quiz from PDF"
                                    )

                                    add_to_chat_history(
                                        "Generated a quiz from PDF",
                                        raw_quiz
                                    )

                                    st.success(
                                        "✅ Quiz generated successfully!"
                                    )

                                    st.info(
                                        "Go to 📝 Interactive Quiz "
                                        "from the sidebar."
                                    )


                            # ==========================================
                            # SUMMARY / EXPLAIN
                            # ==========================================

                            else:

                                if tool == "📝 Summarize":

                                    prompt = f"""
You are an AI study assistant.

Use ONLY this PDF content:

{pdf_text}

Create a clear college-level summary.

Requirements:

- Simple English.
- Headings.
- Bullet points.
- Important concepts.
- Important technical terms.
- No unrelated information.
"""

                                else:

                                    prompt = f"""
You are an AI study assistant.

Use ONLY this PDF content:

{pdf_text}

Explain it to a beginner college student.

Requirements:

- Very simple English.
- Step-by-step explanation.
- Explain important concepts.
- Give simple examples where useful.
- No unrelated information.
"""


                                try:

                                    with st.spinner(
                                        "🤖 Groq is analyzing your PDF..."
                                    ):

                                        answer = ask_ai(
                                            prompt
                                        )


                                    add_to_chat_history(
                                        tool,
                                        answer
                                    )

                                    st.session_state.last_ai_response = answer


                                    if tool == "📝 Summarize":

                                        st.session_state.last_learning_activity = (
                                            "Generated a PDF summary"
                                        )

                                    else:

                                        st.session_state.last_learning_activity = (
                                            "Generated a PDF explanation"
                                        )


                                    st.markdown(
                                        '<div class="result-box">',
                                        unsafe_allow_html=True
                                    )

                                    st.subheader(
                                        "🤖 AI Response"
                                    )

                                    st.markdown(
                                        answer
                                    )

                                    st.markdown(
                                        '</div>',
                                        unsafe_allow_html=True
                                    )


                                    st.download_button(
                                        "📥 Download AI Response",
                                        data=answer,
                                        file_name=
                                        "ai_pdf_study_response.txt",
                                        mime="text/plain",
                                        use_container_width=True
                                    )


                                except Exception as e:

                                    st.error(
                                        "⚠️ Unable to generate "
                                        "a response."
                                    )

                                    with st.expander(
                                        "Technical details"
                                    ):

                                        st.code(
                                            str(e)
                                        )


            except Exception as e:

                st.error(
                    "❌ Unable to read this PDF."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(e)
                    )


# ============================================================
# PAGE 2
# INTERACTIVE QUIZ
# ============================================================

elif page == "📝 Interactive Quiz":

    st.markdown(
        """
<div class="study-card">

<div class="study-title">
📝 Interactive Quiz
</div>

<div class="study-description">
Test your knowledge and calculate your score.
</div>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # NO QUIZ
    # ========================================================

    if not st.session_state.quiz_data:

        st.info(
            "📚 No quiz is available yet."
        )

        st.markdown(
            """
### How to create a quiz

1. Go to **📖 Study Assistant**.
2. Choose **Text Input** or **PDF Upload**.
3. Select **❓ Generate Quiz**.
4. Generate your quiz.
5. Come back to **📝 Interactive Quiz**.
6. Answer all 5 questions.
7. Submit the quiz.
"""
        )


    # ========================================================
    # QUIZ AVAILABLE
    # ========================================================

    else:

        quiz_data = st.session_state.quiz_data

        st.success(
            f"🎯 Quiz ready! You have {len(quiz_data)} questions."
        )


        if not st.session_state.quiz_submitted:

            st.markdown(
                "## 🧠 Answer the Questions"
            )

            answers = {}

            for i, question in enumerate(
                quiz_data
            ):

                st.markdown(
                    f"""
<div class="quiz-card">

<div class="quiz-question">

Question {i + 1}: {question["question"]}

</div>

</div>
""",
                    unsafe_allow_html=True
                )


                options = question["options"]


                selected = st.radio(
                    f"Choose your answer for Question {i + 1}:",
                    [
                        f"A. {options['A']}",
                        f"B. {options['B']}",
                        f"C. {options['C']}",
                        f"D. {options['D']}"
                    ],
                    key=f"quiz_question_{i}"
                )


                answers[i] = selected[0]

                st.markdown("---")


            # =================================================
            # SUBMIT
            # =================================================

            if st.button(
                "✅ Submit Quiz",
                use_container_width=True
            ):

                score = 0

                for i, question in enumerate(
                    quiz_data
                ):

                    if answers[i] == question["answer"]:

                        score += 1


                st.session_state.quiz_score = score

                st.session_state.quiz_answers = answers

                st.session_state.quiz_submitted = True


                # Save to SQLite

                save_quiz_result(
                    "AI Generated Quiz",
                    len(quiz_data),
                    score
                )


                st.session_state.last_learning_activity = (
                    "Completed an interactive quiz"
                )

                st.rerun()


        # ====================================================
        # RESULTS
        # ====================================================

        else:

            score = st.session_state.quiz_score

            total = len(quiz_data)

            percentage = (
                score / total
            ) * 100


            st.markdown(
                f"""
<div class="score-box">

<div>
🎉 Quiz Completed!
</div>

<div class="score-number">
{score} / {total}
</div>

<div>
Score: {percentage:.0f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )


            if percentage >= 80:

                st.success(
                    "🌟 Excellent! You have a strong understanding."
                )

            elif percentage >= 60:

                st.info(
                    "👍 Good job! Review the incorrect answers once."
                )

            else:

                st.warning(
                    "📚 Keep studying and try the quiz again."
                )


            # =================================================
            # ANSWER REVIEW
            # =================================================

            st.markdown(
                "## 📋 Answer Review"
            )


            user_answers = (
                st.session_state.quiz_answers
            )


            for i, question in enumerate(
                quiz_data
            ):

                correct = question["answer"]

                options = question["options"]

                user_answer = user_answers.get(
                    i,
                    "-"
                )


                if user_answer == correct:

                    st.markdown(
                        f"""
<div class="correct-box">

<b>Question {i + 1}</b>

<br><br>

{question["question"]}

<br><br>

Your answer:

<b>{user_answer}. {options[user_answer]}</b>

<br><br>

✅ Correct

</div>
""",
                        unsafe_allow_html=True
                    )


                else:

                    st.markdown(
                        f"""
<div class="wrong-box">

<b>Question {i + 1}</b>

<br><br>

{question["question"]}

<br><br>

Your answer:

<b>{user_answer}. {options.get(user_answer, "Not answered")}</b>

<br><br>

Correct answer:

<b>{correct}. {options[correct]}</b>

</div>
""",
                        unsafe_allow_html=True
                    )


            # =================================================
            # RESTART
            # =================================================

            if st.button(
                "🔄 Take Another Quiz",
                use_container_width=True
            ):

                reset_quiz()

                st.rerun()


            # =================================================
            # DOWNLOAD RESULT
            # =================================================

            result_text = (
                "AI Study Assistant - Quiz Result\n\n"
                f"Score: {score}/{total}\n"
                f"Percentage: {percentage:.0f}%\n\n"
            )


            for i, question in enumerate(
                quiz_data
            ):

                user_answer = user_answers.get(
                    i,
                    "-"
                )

                result_text += (
                    f"Question {i + 1}: "
                    f"{question['question']}\n"
                    f"Your Answer: "
                    f"{user_answer}\n"
                    f"Correct Answer: "
                    f"{question['answer']}\n\n"
                )


            st.download_button(
                "📥 Download Quiz Result",
                data=result_text,
                file_name="quiz_result.txt",
                mime="text/plain",
                use_container_width=True
            )


# ============================================================
# PAGE 3
# QUIZ HISTORY
# ============================================================

elif page == "📚 Quiz History":

    st.markdown(
        """
<div class="study-card">

<div class="study-title">
📚 Quiz History
</div>

<div class="study-description">
View all completed quiz attempts stored in SQLite.
</div>

</div>
""",
        unsafe_allow_html=True
    )


    history = get_quiz_history()


    if not history:

        st.info(
            "📚 No quiz attempts yet."
        )

        st.markdown(
            """
### How to create quiz history

1. Go to **📖 Study Assistant**.
2. Generate a quiz.
3. Open **📝 Interactive Quiz**.
4. Complete the quiz.
5. Submit your answers.
6. Your result will automatically be saved in SQLite.
"""
        )


    else:

        st.markdown(
            "## 📖 Previous Quiz Attempts"
        )


        for item in history:

            (
                quiz_id,
                quiz_name,
                total_questions,
                correct_answers,
                accuracy,
                completed_at
            ) = item


            st.markdown(
                f"""
<div class="history-card">

<div class="history-title">

📘 Quiz Attempt #{quiz_id}

</div>

<div class="history-detail">

📝 Quiz:
<b>{quiz_name}</b>

<br><br>

❓ Questions:
<b>{total_questions}</b>

<br><br>

✅ Correct:
<b>{correct_answers}</b>

<br><br>

🎯 Accuracy:
<b>{accuracy:.0f}%</b>

<br><br>

🕒 Completed:
<b>{completed_at}</b>

</div>

</div>
""",
                unsafe_allow_html=True
            )


        st.markdown("---")


        if st.button(
            "🗑️ Clear Quiz History",
            use_container_width=True
        ):

            clear_database_history()

            st.success(
                "✅ Quiz history cleared from SQLite."
            )

            st.rerun()


# ============================================================
# PAGE 4
# STUDY DASHBOARD
# ============================================================

elif page == "📊 Study Dashboard":

    st.markdown(
        """
<div class="study-card">

<div class="study-title">
📊 Study Dashboard
</div>

<div class="study-description">
Track your learning activity, quiz performance
and PDF-based study activity.
</div>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # LOAD DATABASE HISTORY
    # ========================================================

    history = get_quiz_history()


    quizzes_completed = len(history)


    questions_answered = sum(
        item[2]
        for item in history
    )


    correct_answers = sum(
        item[3]
        for item in history
    )


    if questions_answered > 0:

        overall_accuracy = (
            correct_answers /
            questions_answered
        ) * 100

    else:

        overall_accuracy = 0


    # ========================================================
    # TOP STATISTICS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="dashboard-number">
{quizzes_completed}
</div>

<div class="dashboard-label">
📝 Quizzes Completed
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="dashboard-number">
{questions_answered}
</div>

<div class="dashboard-label">
❓ Questions Answered
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="dashboard-number">
{correct_answers}
</div>

<div class="dashboard-label">
✅ Correct Answers
</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col4:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="dashboard-number">
{overall_accuracy:.0f}%
</div>

<div class="dashboard-label">
🎯 Overall Accuracy
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # PDF INFORMATION
    # ========================================================

    st.markdown(
        "## 📄 Current PDF Activity"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="history-title">
📄 Current PDF
</div>

<div class="history-detail">

{st.session_state.current_pdf_name}

<br><br>

📑 Pages:

<b>{st.session_state.current_pdf_pages}</b>

</div>

</div>
""",
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            f"""
<div class="dashboard-card">

<div class="history-title">
🔎 RAG Information
</div>

<div class="history-detail">

📚 Number of RAG chunks:

<b>{st.session_state.rag_chunk_count}</b>

</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # QUIZ HISTORY
    # ========================================================

    st.markdown(
        "## 📚 Quiz History"
    )


    if not history:

        st.info(
            "No completed quizzes yet."
        )

    else:

        for item in history:

            (
                quiz_id,
                quiz_name,
                total_questions,
                correct_answers_item,
                accuracy,
                completed_at
            ) = item


            st.markdown(
                f"""
<div class="history-card">

<div class="history-title">

📘 Quiz Attempt #{quiz_id}

</div>

<div class="history-detail">

📝 Quiz:
<b>{quiz_name}</b>

<br><br>

❓ Questions:
<b>{total_questions}</b>

<br><br>

✅ Correct:
<b>{correct_answers_item}</b>

<br><br>

🎯 Accuracy:
<b>{accuracy:.0f}%</b>

<br><br>

🕒 Completed:
<b>{completed_at}</b>

</div>

</div>
""",
                unsafe_allow_html=True
            )


    # ========================================================
    # LEARNING SUMMARY
    # ========================================================

    st.markdown(
        "## 🧠 Learning Summary"
    )


    st.markdown(
        f"""
<div class="dashboard-card">

<div class="history-title">

📈 Your Current Learning Activity

</div>

<div class="history-detail">

🔹 You have completed
<b>{quizzes_completed}</b> quiz attempt(s).

<br><br>

🔹 You answered
<b>{questions_answered}</b> question(s).

<br><br>

🔹 You answered
<b>{correct_answers}</b> question(s) correctly.

<br><br>

🔹 Your overall quiz accuracy is
<b>{overall_accuracy:.0f}%</b>.

<br><br>

🔹 Current PDF:
<b>{st.session_state.current_pdf_name}</b>

<br><br>

🔹 PDF pages:
<b>{st.session_state.current_pdf_pages}</b>

<br><br>

🔹 RAG chunks created:
<b>{st.session_state.rag_chunk_count}</b>

<br><br>

🔹 Latest activity:
<b>{st.session_state.last_learning_activity}</b>

</div>

</div>
""",
        unsafe_allow_html=True
    )


    # ========================================================
    # STUDY ACTIVITY
    # ========================================================

    st.markdown(
        "## 🎯 Study Activity"
    )


    if quizzes_completed == 0:

        st.info(
            "📚 Start by generating and completing your first quiz."
        )

    elif overall_accuracy >= 80:

        st.success(
            "🌟 Your current quiz accuracy is 80% or higher."
        )

    elif overall_accuracy >= 60:

        st.info(
            "👍 Review incorrect answers to strengthen your understanding."
        )

    else:

        st.warning(
            "📖 Continue studying and use the quiz review to identify topics that need more practice."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">

<hr>

<p>
🎓 <b>AI Study Assistant</b>
</p>

<p>
Built using Python • Streamlit • Groq AI •
FAISS • Sentence Transformers • SQLite
</p>

<p>
Study smarter. Learn better. 🚀
</p>

</div>
""",
    unsafe_allow_html=True
)