import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD GEMINI API KEY
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0d1117;
}


/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #171b27;
}


/* Hero */
.hero {
    background: linear-gradient(135deg, #667eea, #764ba2);
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


/* White study card */
.study-card {
    background-color: #ffffff;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.20);
}

.study-title {
    color: #172033;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}

.study-description {
    color: #4b5563;
    font-size: 17px;
}


/* Text area */
textarea {
    color: #172033 !important;
    background-color: #ffffff !important;
}

textarea::placeholder {
    color: #718096 !important;
    opacity: 1 !important;
}


/* Generate button */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
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


/* Download button */
.stDownloadButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 600;
}


/* Result box */
.result-box {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #667eea;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.15);
    margin-top: 20px;
}

.result-box p,
.result-box li,
.result-box strong {
    color: #172033 !important;
}


/* Footer */
.footer {
    text-align: center;
    color: #9ca3af;
    padding: 30px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🎓 AI Study Assistant")

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Input Mode")

input_mode = st.sidebar.radio(
    "Choose how you want to provide content:",
    [
        "✍️ Text Input",
        "📄 PDF Upload"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ Study Tool")

tool = st.sidebar.radio(
    "Choose an AI tool:",
    [
        "📝 Summarize",
        "💡 Explain",
        "❓ Generate Quiz"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "This application uses Gemini AI to help students "
    "summarize notes, understand concepts and generate quizzes."
)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🎓 AI Study Assistant</div>'
    '<div class="hero-subtitle">'
    'Study smarter with AI-powered summaries, explanations and quizzes.'
    '</div>'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TEXT INPUT MODE
# =========================================================

if input_mode == "✍️ Text Input":

    # -----------------------------------------------------
    # STUDY CONTENT CARD
    # -----------------------------------------------------

    st.markdown(
        '<div class="study-card">'
        '<div class="study-title">'
        '✏️ Enter Your Study Content'
        '</div>'
        '<div class="study-description">'
        'Paste your notes, topic, question or study material below.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # TEXT INPUT
    # -----------------------------------------------------

    text = st.text_area(
        "Enter your content:",
        height=250,
        placeholder="Example: Explain Operating System in simple terms..."
    )


    # -----------------------------------------------------
    # GENERATE BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚀 Generate AI Response",
        use_container_width=True
    ):

        # Empty input validation
        if not text.strip():

            st.warning(
                "⚠️ Please enter some content before generating a response."
            )

        else:

            # -------------------------------------------------
            # SUMMARIZE
            # -------------------------------------------------

            if tool == "📝 Summarize":

                prompt = f"""
You are an AI study assistant.

Summarize the following study material for a college student.

Requirements:
- Use simple English.
- Keep the important points.
- Use headings and bullet points.
- Make it easy to study and revise.
- Do not remove important technical terms.

Study material:

{text}
"""


            # -------------------------------------------------
            # EXPLAIN
            # -------------------------------------------------

            elif tool == "💡 Explain":

                prompt = f"""
You are an AI study assistant.

Explain the following topic to a beginner college student.

Requirements:
- Use very simple English.
- Explain step by step.
- Give a simple real-world example if useful.
- Highlight important points.
- Avoid unnecessarily complicated language.

Topic:

{text}
"""


            # -------------------------------------------------
            # QUIZ
            # -------------------------------------------------

            else:

                prompt = f"""
You are an AI study assistant.

Create a quiz from the following study material.

Requirements:
- Create 5 multiple-choice questions.
- Give 4 options for each question.
- Clearly show the correct answer.
- Give a short explanation for each answer.
- Questions should be useful for college exam preparation.

Study material:

{text}
"""


            # -------------------------------------------------
            # GEMINI API
            # -------------------------------------------------

            try:

                with st.spinner(
                    "🤖 AI is generating your answer..."
                ):

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=prompt
                    )

                    answer = interaction.output_text


                # -------------------------------------------------
                # DISPLAY RESPONSE
                # -------------------------------------------------

                if not answer or not answer.strip():

                    st.warning(
                        "⚠️ The AI did not generate a response. "
                        "Please try again."
                    )

                else:

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    st.subheader("🤖 AI Response")

                    st.markdown(answer)

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


                    # Download response
                    st.download_button(
                        "📥 Download AI Response",
                        data=answer,
                        file_name="ai_study_response.txt",
                        mime="text/plain",
                        use_container_width=True
                    )


            except Exception as e:

                st.error(
                    "⚠️ Unable to generate a response right now. "
                    "Please try again."
                )

                with st.expander("Technical details"):
                    st.code(str(e))


# =========================================================
# PDF UPLOAD MODE
# =========================================================

else:

    # -----------------------------------------------------
    # PDF CARD
    # -----------------------------------------------------

    st.markdown(
        '<div class="study-card">'
        '<div class="study-title">'
        '📄 Upload Your Study PDF'
        '</div>'
        '<div class="study-description">'
        'Upload a PDF containing your study material and let AI '
        'summarize, explain or create a quiz from it.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # PDF UPLOADER
    # -----------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a PDF file:",
        type=["pdf"]
    )


    if uploaded_file is not None:

        try:

            # Read PDF
            reader = PdfReader(uploaded_file)

            pdf_text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pdf_text += page_text + "\n"


            # -------------------------------------------------
            # CHECK PDF
            # -------------------------------------------------

            if not pdf_text.strip():

                st.warning(
                    "⚠️ Could not extract text from this PDF."
                )

            else:

                st.success(
                    f"✅ PDF uploaded successfully! "
                    f"Pages: {len(reader.pages)}"
                )


                # -------------------------------------------------
                # LIMIT TEXT
                # -------------------------------------------------

                MAX_CHARS = 50000

                if len(pdf_text) > MAX_CHARS:

                    pdf_text = pdf_text[:MAX_CHARS]

                    st.info(
                        "ℹ️ The PDF is large, so only the first "
                        "50,000 characters will be processed."
                    )


                # -------------------------------------------------
                # PREVIEW
                # -------------------------------------------------

                with st.expander("👀 Preview Extracted Text"):

                    st.text(pdf_text[:3000])


                # -------------------------------------------------
                # GENERATE BUTTON
                # -------------------------------------------------

                if st.button(
                    "🚀 Generate AI Response",
                    use_container_width=True
                ):

                    # -------------------------------------------------
                    # SUMMARIZE PDF
                    # -------------------------------------------------

                    if tool == "📝 Summarize":

                        prompt = f"""
You are an AI study assistant.

Summarize the following PDF study material for a college student.

Requirements:
- Use simple English.
- Keep the important concepts.
- Use headings and bullet points.
- Make it easy to study and revise.
- Do not remove important technical terms.

PDF study material:

{pdf_text}
"""


                    # -------------------------------------------------
                    # EXPLAIN PDF
                    # -------------------------------------------------

                    elif tool == "💡 Explain":

                        prompt = f"""
You are an AI study assistant.

Explain the following PDF study material to a beginner college student.

Requirements:
- Use very simple English.
- Explain important concepts step by step.
- Give simple examples where useful.
- Highlight important points.
- Make the explanation easy to understand.

PDF study material:

{pdf_text}
"""


                    # -------------------------------------------------
                    # QUIZ PDF
                    # -------------------------------------------------

                    else:

                        prompt = f"""
You are an AI study assistant.

Create a quiz from the following PDF study material.

Requirements:
- Create 5 multiple-choice questions.
- Give 4 options for each question.
- Clearly show the correct answer.
- Give a short explanation for each answer.
- Questions should be useful for college exam preparation.

PDF study material:

{pdf_text}
"""


                    # -------------------------------------------------
                    # GEMINI API
                    # -------------------------------------------------

                    try:

                        with st.spinner(
                            "🤖 AI is analyzing your PDF..."
                        ):

                            interaction = client.interactions.create(
                                model="gemini-3.6-flash",
                                input=prompt
                            )

                            answer = interaction.output_text


                        # -------------------------------------------------
                        # DISPLAY RESPONSE
                        # -------------------------------------------------

                        if not answer or not answer.strip():

                            st.warning(
                                "⚠️ The AI did not generate a response. "
                                "Please try again."
                            )

                        else:

                            st.markdown(
                                '<div class="result-box">',
                                unsafe_allow_html=True
                            )

                            st.subheader("🤖 AI Response")

                            st.markdown(answer)

                            st.markdown(
                                '</div>',
                                unsafe_allow_html=True
                            )


                            # Download response
                            st.download_button(
                                "📥 Download AI Response",
                                data=answer,
                                file_name="ai_pdf_study_response.txt",
                                mime="text/plain",
                                use_container_width=True
                            )


                    except Exception as e:

                        st.error(
                            "⚠️ Unable to generate a response right now. "
                            "Please try again."
                        )

                        with st.expander("Technical details"):
                            st.code(str(e))


        except Exception as e:

            st.error(
                "❌ Unable to read this PDF. "
                "Please make sure the file is a valid PDF."
            )

            with st.expander("Technical details"):
                st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '<hr>'
    '<p>🎓 <b>AI Study Assistant</b></p>'
    '<p>Built using Python • Streamlit • Gemini AI</p>'
    '<p>Study smarter. Learn better. 🚀</p>'
    '</div>',
    unsafe_allow_html=True
)