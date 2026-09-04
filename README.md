# 🤖 AI Customer Support Assistant

A real-time, AI-powered customer support agent dashboard built with Python, Streamlit, Google Gemini AI (`gemini-3.6-flash`), and Hugging Face Transformers (`bart-large-mnli` and `distilbert`).

---

## 🌟 Key Features

- **📊 Executive Dashboard**: High-level interaction metrics, escalation counts, and real-time risk summaries.
- **💬 Real-Time Conversation Workspace**:
  - **Risk Analysis**: Automatic detection of **Sentiment** (Positive/Neutral/Negative), **Urgency** (Low/Medium/High), and **Escalation Risk** (Low/Medium/High).
  - **Intent Classification**: Machine learning intent detection (complaint, query, purchase, technical issue, feedback).
  - **Key Issue Extraction**: Automatic summary of customer concerns.
  - **AI Suggested Reply**: Gemini AI recommended responses with one-click copy to agent input.
- **🎯 Agent AI Coaching**:
  - Real-time response scoring for **Tone**, **Empathy**, and **Clarity** (1–10).
  - Actionable coaching tips and recommendations.
- **📈 Performance Analytics**:
  - Interactive sentiment distribution charts and coaching score trend lines.

---

## 🛠️ Project Structure

```text
AI-Customer-Support-Assistant/
│
├── backend/
│   ├── ai_coach.py          # Gemini AI risk analysis & coaching evaluation logic
│   ├── coaching_session.py  # RealTimeCoachingSession manager & state tracker
│   ├── intent.py            # Hugging Face BART zero-shot intent classifier
│   └── sentiment.py         # Hugging Face DistilBERT sentiment analyzer
│
├── frontend/
│   └── app.py               # Main Streamlit SaaS Dashboard UI
│
├── .env.example             # Environment variable template
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Customer-Support-Assistant.git
cd AI-Customer-Support-Assistant
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory and add your Gemini API Key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run frontend/app.py
```

Open your browser at `http://localhost:8501`.
