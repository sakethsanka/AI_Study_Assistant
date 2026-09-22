import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
import os


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# LOAD GEMINI API KEY
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Gemini API key not found. Please check your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)


# --------------------------------------------------
# WEBSITE THEME
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.footer {
    text-align: center;
    padding: 25px;
    color: #777;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 📚 AI Study Assistant")

    st.write(
        "Your personal AI-powered study companion."
    )

    st.divider()

    st.markdown("### 📥 Input Mode")

    input_mode = st.radio(
        "Choose how you want to study:",
        [
            "✍️ Text Input",
            "📄 PDF Upload"
        ]
    )

    st.divider()

    st.markdown("### 🛠️ Study Tools")

    option = st.radio(
        "Choose a tool:",
        [
            "📝 Summarize",
            "💡 Explain",
            "❓ Generate Quiz"
        ]
    )

    st.divider()

    st.markdown("### ✨ Features")

    st.write("📝 Summarize study material")
    st.write("💡 Explain difficult concepts")
    st.write("❓ Generate practice quizzes")
    st.write("📄 Study from PDF documents")
    st.write("🤖 Powered by Gemini AI")

    st.divider()

    st.caption(
        "Built with Python + Streamlit + Gemini AI"
    )


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown("""
<div class="hero">

<h1>📚 AI Study Assistant</h1>

<p>
Learn smarter • Understand faster • Practice better
</p>

</div>
""", unsafe_allow_html=True)


st.markdown(
    "### 👋 Welcome to your AI Study Assistant"
)

st.write(
    "Choose a study tool, provide your notes or PDF, "
    "and let AI help you learn."
)


# ==================================================
# TEXT INPUT MODE
# ==================================================

if input_mode == "✍️ Text Input":

    st.markdown("### ✍️ Enter Your Study Material")

    if option == "📝 Summarize":

        text = st.text_area(
            "Paste your notes here:",
            height=250,
            placeholder="Paste your study notes here..."
        )

        button_text = "📝 Summarize Notes"

    elif option == "💡 Explain":

        text = st.text_area(
            "Enter the concept you want explained:",
            height=250,
            placeholder="Example: Explain Operating System..."
        )

        button_text = "💡 Explain Concept"

    else:

        text = st.text_area(
            "Enter the topic or notes for the quiz:",
            height=250,
            placeholder="Example: Operating System, DBMS, Algorithms..."
        )

        button_text = "❓ Generate Quiz"


    # --------------------------------------------------
    # GENERATE BUTTON
    # --------------------------------------------------

    if st.button(
        button_text,
        use_container_width=True
    ):

        if not text.strip():

            st.warning(
                "⚠️ Please enter some text before generating an answer."
            )

        else:

            # ------------------------------------------
            # CREATE PROMPT
            # ------------------------------------------

            if option == "📝 Summarize":

                prompt = f"""
You are a helpful AI study assistant.

Summarize the following study notes.

Requirements:
- Use simple English.
- Keep the important points.
- Use headings and bullet points.
- Make it easy for a college student to study.
- Do not add unrelated information.

Study Notes:

{text}
"""

            elif option == "💡 Explain":

                prompt = f"""
You are a beginner-friendly AI teacher.

Explain the following concept clearly.

Requirements:
- Start with a simple definition.
- Explain step by step.
- Give a simple real-world example.
- Use simple English.
- Mention important points at the end.

Concept:

{text}
"""

            else:

                prompt = f"""
You are an AI quiz generator for college students.

Create a quiz based on the following topic or notes.

Requirements:
- Create 5 multiple-choice questions.
- Give 4 options for each question.
- Clearly show the correct answer.
- Give a short explanation for each answer.
- Questions should be based only on the provided material.

Study Material:

{text}
"""


            # ------------------------------------------
            # GEMINI API
            # ------------------------------------------

            try:

                with st.spinner(
                    "🤖 AI is generating your answer..."
                ):

                    interaction = client.interactions.create(
                        model="gemini-3.6-flash",
                        input=prompt
                    )

                    answer = interaction.output_text


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


            except Exception as e:

                st.error(
                    "⚠️ Unable to generate a response right now. "
                    "Please try again."
                )

                with st.expander("Technical details"):

                    st.code(str(e))


# ==================================================
# PDF INPUT MODE
# ==================================================

else:

    st.markdown("### 📄 Upload Your Study PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your notes, textbook chapter, or study material."
    )


    # --------------------------------------------------
    # IF PDF IS UPLOADED
    # --------------------------------------------------

    if uploaded_file is not None:

        try:

            # ------------------------------------------
            # READ PDF
            # ------------------------------------------

            pdf_reader = PdfReader(uploaded_file)

            number_of_pages = len(pdf_reader.pages)

            st.success(
                f"✅ PDF uploaded successfully! "
                f"Pages: {number_of_pages}"
            )


            # ------------------------------------------
            # EXTRACT TEXT
            # ------------------------------------------

            pdf_text = ""

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:

                    pdf_text += page_text + "\n"


            # ------------------------------------------
            # CHECK PDF TEXT
            # ------------------------------------------

            if not pdf_text.strip():

                st.error(
                    "❌ Could not extract text from this PDF."
                )

                st.info(
                    "Please try a PDF that contains selectable text."
                )

                st.stop()


            # ------------------------------------------
            # SHOW PDF INFORMATION
            # ------------------------------------------

            st.info(
                f"📖 Extracted approximately "
                f"{len(pdf_text):,} characters from the PDF."
            )


            # --------------------------------------------------
            # LIMIT VERY LARGE PDF
            # --------------------------------------------------

            MAX_CHARS = 50000

            if len(pdf_text) > MAX_CHARS:

                st.warning(
                    "⚠️ This PDF is large. "
                    "Only the first 50,000 characters will be "
                    "used to keep the AI request manageable."
                )

                pdf_text = pdf_text[:MAX_CHARS]


            # ------------------------------------------
            # PDF PREVIEW
            # ------------------------------------------

            with st.expander("👀 Preview extracted PDF text"):

                st.text(pdf_text[:3000])

                if len(pdf_text) > 3000:

                    st.write(
                        "... Preview shortened ..."
                    )


            # ------------------------------------------
            # PDF ACTION
            # ------------------------------------------

            st.markdown("### 🛠️ Choose what AI should do")

            if option == "📝 Summarize":

                st.write(
                    "Create a clear summary of the uploaded PDF."
                )

                pdf_button_text = "📝 Summarize PDF"

            elif option == "💡 Explain":

                st.write(
                    "Explain the important concepts from the PDF."
                )

                pdf_button_text = "💡 Explain PDF"

            else:

                st.write(
                    "Create a practice quiz from the PDF."
                )

                pdf_button_text = "❓ Generate Quiz from PDF"


            # ------------------------------------------
            # PDF GENERATE BUTTON
            # ------------------------------------------

            if st.button(
                pdf_button_text,
                use_container_width=True
            ):


                # --------------------------------------
                # CREATE PDF PROMPT
                # --------------------------------------

                if option == "📝 Summarize":

                    prompt = f"""
You are a helpful AI study assistant.

The user uploaded a study PDF.

Create a clear and useful summary of the
PDF content.

Requirements:
- Use simple English.
- Organize the answer with headings.
- Use bullet points.
- Include important definitions.
- Include important algorithms or concepts.
- Include important examples when useful.
- Do not add unrelated information.
- Make the result useful for exam preparation.

PDF CONTENT:

{pdf_text}
"""


                elif option == "💡 Explain":

                    prompt = f"""
You are a beginner-friendly AI teacher.

The following content was extracted from
a student's study PDF.

Explain the important concepts clearly.

Requirements:
- Use simple English.
- Start with the important concepts.
- Explain each concept step by step.
- Give simple examples where useful.
- Explain difficult terms.
- Mention important points for exams.
- Base the explanation on the provided PDF.

PDF CONTENT:

{pdf_text}
"""


                else:

                    prompt = f"""
You are an AI quiz generator for college students.

Create a quiz using the uploaded PDF content.

Requirements:
- Create 5 multiple-choice questions.
- Each question must have 4 options.
- Clearly show the correct answer.
- Give a short explanation for each answer.
- Questions must be based on the PDF.
- Do not create questions about information
  that is not present in the PDF.
- Use a mixture of easy and medium questions.

PDF CONTENT:

{pdf_text}
"""


                # --------------------------------------
                # CALL GEMINI
                # --------------------------------------

                try:

                    with st.spinner(
                        "🤖 Reading the PDF and generating your answer..."
                    ):

                        interaction = client.interactions.create(
                            model="gemini-3.6-flash",
                            input=prompt
                        )

                        answer = interaction.output_text


                    # ----------------------------------
                    # DISPLAY RESULT
                    # ----------------------------------

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


                except Exception as e:

                    st.error(
                        "⚠️ Unable to generate a response right now. "
                        "Please try again."
                    )

                    with st.expander("Technical details"):

                        st.code(str(e))


        except Exception as e:

            st.error(
                "❌ There was a problem reading this PDF."
            )

            with st.expander("Technical details"):

                st.code(str(e))


    else:

        st.info(
            "📄 Upload a PDF above to start studying from your document."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("""
<div class="footer">

<hr>

📚 <b>AI Study Assistant</b>

<br>

Built using Python • Streamlit • Gemini AI

<br><br>

Learn smarter. Understand faster. Practice better. 🚀

</div>
""", unsafe_allow_html=True)