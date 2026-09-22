# 🎓 AI Study Assistant

An AI-powered study assistant designed to help college students understand and revise their study materials using Gemini AI.

The application provides three main study tools:

- 📝 Summarize study material
- 💡 Explain concepts in simple language
- ❓ Generate multiple-choice quizzes

The application also supports PDF upload, allowing students to use their study notes directly from PDF files.

---

## 📌 Problem Statement

College students often have large amounts of study material to read and revise.

Understanding lengthy notes, preparing short summaries, and creating practice questions manually can take a lot of time.

The **AI Study Assistant** provides a simple application where students can enter their study material or upload a PDF and use AI to:

- Understand difficult concepts
- Quickly summarize notes
- Practice using automatically generated quizzes

---

## 🎯 Objectives

The main objectives of this project are:

1. Build a simple AI-powered student utility application.
2. Integrate Gemini AI into a usable application.
3. Help students summarize study materials.
4. Explain difficult concepts using simple language.
5. Generate practice quizzes from study material.
6. Support PDF-based study material.
7. Provide validation for empty input.
8. Handle API errors properly.
9. Allow users to download generated AI responses.

---

## 🚀 Features

### 1. 📝 Summarize

Students can enter notes or study material and generate a concise summary.

The AI:

- Identifies important points
- Uses headings and bullet points
- Keeps important technical terms
- Makes the content easier to revise

---

### 2. 💡 Explain

Students can enter a topic or question and ask the AI to explain it.

The AI:

- Uses simple English
- Explains concepts step by step
- Provides examples when useful
- Highlights important points

---

### 3. ❓ Generate Quiz

Students can generate a quiz from their study material.

The application generates:

- 5 multiple-choice questions
- 4 options for each question
- Correct answers
- Short explanations

This can be used for exam preparation and self-testing.

---

### 4. 📄 PDF Upload

Students can upload a PDF containing study material.

The application:

1. Accepts the PDF file.
2. Extracts text from the PDF.
3. Sends the extracted content to Gemini AI.
4. Generates a summary, explanation, or quiz.

---

### 5. 📥 Download AI Response

Generated responses can be downloaded as a `.txt` file for later study or revision.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | User interface |
| Gemini AI | AI-powered responses |
| Google GenAI SDK | Gemini API integration |
| python-dotenv | Environment variable management |
| pypdf | PDF text extraction |

---

## 🏗️ Application Workflow

```text
                ┌─────────────────────┐
                │   Student opens app │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    Select Input     │
                │                     │
                │  Text / PDF Upload  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Select Study Tool   │
                │                     │
                │ Summarize           │
                │ Explain             │
                │ Generate Quiz       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Create Prompt     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     Gemini AI       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Display AI Response │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Download Response   │
                └─────────────────────┘