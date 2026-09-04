import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path for backend imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING (Render UI immediately)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS Dashboard Styling
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .panel-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .panel-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 14px;
        border-bottom: 1px solid #334155;
        padding-bottom: 8px;
    }

    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-green { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-yellow { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-red { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    .badge-blue { background-color: #1e3a8a; color: #60a5fa; border: 1px solid #2563eb; }
    .badge-gray { background-color: #334155; color: #cbd5e1; border: 1px solid #475569; }

    .chat-msg {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .chat-customer {
        background-color: #1e3a8a22;
        border-left: 4px solid #3b82f6;
        color: #e2e8f0;
    }
    .chat-agent {
        background-color: #064e3b22;
        border-left: 4px solid #10b981;
        color: #e2e8f0;
    }
    .chat-meta {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 4px;
        text-transform: uppercase;
    }

    .coaching-tip-card {
        background-color: #312e8122;
        border: 1px solid #4f46e5;
        border-radius: 10px;
        padding: 16px;
        color: #c7d2fe;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LAZY-LOAD BACKEND WITH STREAMLIT CACHING (Shows spinner on startup)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⏳ Loading AI Models (BART & DistilBERT)...")
def get_backend_session_cls():
    try:
        from backend.coaching_session import RealTimeCoachingSession
        return RealTimeCoachingSession, None
    except Exception as e:
        return None, str(e)

RealTimeCoachingSession, BACKEND_ERROR = get_backend_session_cls()
BACKEND_AVAILABLE = RealTimeCoachingSession is not None

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE
# -----------------------------------------------------------------------------
if "session" not in st.session_state and BACKEND_AVAILABLE:
    try:
        st.session_state.session = RealTimeCoachingSession()
    except Exception as e:
        st.session_state.session_init_error = str(e)
        st.session_state.session = None

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "last_feedback" not in st.session_state:
    st.session_state.last_feedback = None

if "suggested_reply" not in st.session_state:
    st.session_state.suggested_reply = ""

if "agent_response_field" not in st.session_state:
    st.session_state["agent_response_field"] = ""

if "customer_input_field" not in st.session_state:
    st.session_state["customer_input_field"] = ""

if "analyses_history" not in st.session_state:
    st.session_state.analyses_history = []

if "feedback_history" not in st.session_state:
    st.session_state.feedback_history = []

# -----------------------------------------------------------------------------
# CALLBACK FUNCTIONS (Execute before widget instantiation to prevent errors)
# -----------------------------------------------------------------------------
def reset_conversation_callback():
    if BACKEND_AVAILABLE:
        try:
            st.session_state.session = RealTimeCoachingSession()
        except Exception:
            st.session_state.session = None
    st.session_state.last_analysis = None
    st.session_state.last_feedback = None
    st.session_state.suggested_reply = ""
    st.session_state["agent_response_field"] = ""
    st.session_state["customer_input_field"] = ""
    st.session_state.analyses_history = []
    st.session_state.feedback_history = []
    st.toast("New conversation session started!", icon="✨")

def use_suggestion_callback():
    st.session_state["agent_response_field"] = st.session_state.suggested_reply
    st.toast("Suggestion copied to agent response input!", icon="📋")

def clear_inputs_callback():
    st.session_state["agent_response_field"] = ""
    st.session_state["customer_input_field"] = ""

def evaluate_response_callback():
    agent_reply = st.session_state.get("agent_response_field", "").strip()
    if not agent_reply:
        st.toast("Please enter an agent response to evaluate.", icon="⚠️")
        return
    if st.session_state.get("session") is None:
        st.toast("Backend session unavailable.", icon="❌")
        return
    if not st.session_state.session.last_customer_message:
        st.toast("Please process a customer message first.", icon="⚠️")
        return

    try:
        feedback = st.session_state.session.on_agent_message(agent_reply)
        st.session_state.last_feedback = feedback
        st.session_state.feedback_history.append(feedback)
        st.session_state["agent_response_field"] = ""
        st.toast("Agent response evaluated!", icon="⭐")
    except Exception as ex:
        st.toast(f"Error evaluating response: {ex}", icon="❌")

# Helper for status badges
def render_badge(text: str, category: str):
    text_lower = text.lower()
    if text_lower in ["low", "positive"]:
        cls = "badge-green"
    elif text_lower in ["medium", "neutral"]:
        cls = "badge-yellow"
    elif text_lower in ["high", "negative"]:
        cls = "badge-red"
    else:
        cls = "badge-blue"
    return f'<span class="badge {cls}">{text.upper()}</span>'

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Customer Support AI")
    st.markdown("<p style='color:#94a3b8; font-size:0.85rem;'>Real-time Agent Coaching & Analytics</p>", unsafe_allow_html=True)
    st.divider()

    nav_choice = st.radio(
        "Navigation",
        options=["📊 Dashboard", "💬 Conversation", "🎯 AI Coaching", "📈 Analytics"],
        index=1,
        key="nav_selection"
    )

    st.divider()

    st.button("➕ New Conversation", use_container_width=True, on_click=reset_conversation_callback, type="secondary")

    st.divider()
    
    # Session status indicator
    if hasattr(st.session_state, "session_init_error"):
        st.error(f"Initialization Error: {st.session_state.session_init_error}")
    elif BACKEND_AVAILABLE and st.session_state.get("session") is not None:
        st.caption("🟢 Backend Connected (Gemini 3.6 Flash)")
    else:
        st.caption("🔴 Backend Unavailable")
        if BACKEND_ERROR:
            st.error(BACKEND_ERROR)

# -----------------------------------------------------------------------------
# BACKEND CHECK & WARN
# -----------------------------------------------------------------------------
if not BACKEND_AVAILABLE or st.session_state.get("session") is None:
    st.warning("⚠️ Could not connect to backend models. Please verify your GEMINI_API_KEY in `.env` and required packages.")
    if BACKEND_ERROR:
        st.error(f"Error details: {BACKEND_ERROR}")

# -----------------------------------------------------------------------------
# VIEW 1: DASHBOARD
# -----------------------------------------------------------------------------
if nav_choice == "📊 Dashboard":
    st.markdown('<div class="main-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Overview of real-time customer interactions, coaching performance, and risk metrics.</div>', unsafe_allow_html=True)

    # Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    
    total_messages = len(st.session_state.session.state.history) if (st.session_state.get("session") and st.session_state.session.state) else 0
    escalations = sum(1 for a in st.session_state.analyses_history if a.get("escalation_risk", "").lower() == "high")
    
    fb_list = st.session_state.feedback_history
    avg_tone = round(sum(f["tone_score"] for f in fb_list) / len(fb_list), 1) if fb_list else 0.0
    avg_emp = round(sum(f["empathy_score"] for f in fb_list) / len(fb_list), 1) if fb_list else 0.0
    avg_clar = round(sum(f["clarity_score"] for f in fb_list) / len(fb_list), 1) if fb_list else 0.0
    overall_score = round((avg_tone + avg_emp + avg_clar) / 3, 1) if fb_list else 0.0

    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_messages}</div><div class="metric-lbl">Total Session Messages</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{escalations}</div><div class="metric-lbl">High Risk Escalations</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{overall_score} / 10</div><div class="metric-lbl">Avg Response Score</div></div>', unsafe_allow_html=True)
    with m4:
        last_intent = st.session_state.last_analysis["intent"].title() if st.session_state.last_analysis else "None"
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.4rem;">{last_intent}</div><div class="metric-lbl">Latest Detected Intent</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        st.markdown('<div class="panel-card"><div class="panel-title">💬 Current Conversation Snapshot</div>', unsafe_allow_html=True)
        if st.session_state.get("session") and st.session_state.session.state.history:
            for msg in st.session_state.session.state.history[-4:]:
                role_cls = "chat-customer" if msg.speaker.lower() == "customer" else "chat-agent"
                role_name = "👤 Customer" if msg.speaker.lower() == "customer" else "🎧 Support Agent"
                st.markdown(f'''
                <div class="chat-msg {role_cls}">
                    <div class="chat-meta">{role_name}</div>
                    {msg.text}
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No active messages in the current session. Go to the Conversation tab to begin!")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="panel-card"><div class="panel-title">🛡️ Risk & Sentiment Summary</div>', unsafe_allow_html=True)
        if st.session_state.last_analysis:
            an = st.session_state.last_analysis
            st.write("**Sentiment:**", unsafe_allow_html=True)
            st.markdown(render_badge(an.get("sentiment", "N/A"), "sentiment"), unsafe_allow_html=True)
            st.write("")
            st.write("**Urgency:**", unsafe_allow_html=True)
            st.markdown(render_badge(an.get("urgency", "N/A"), "urgency"), unsafe_allow_html=True)
            st.write("")
            st.write("**Escalation Risk:**", unsafe_allow_html=True)
            st.markdown(render_badge(an.get("escalation_risk", "N/A"), "escalation"), unsafe_allow_html=True)
            st.write("")
            st.write("**Key Issue:**")
            st.info(an.get("key_issue", "No issue recorded."))
        else:
            st.caption("No customer message analyzed yet.")
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 2: CONVERSATION (MAIN WORKSPACE)
# -----------------------------------------------------------------------------
elif nav_choice == "💬 Conversation":
    st.markdown('<div class="main-header">💬 Real-Time Conversation Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Process customer messages, view AI risk analysis, get suggested responses, and evaluate agent replies.</div>', unsafe_allow_html=True)

    col_chat, col_ai = st.columns([1.6, 1.2])

    with col_chat:
        # Panel: Customer Input
        st.markdown('<div class="panel-card"><div class="panel-title">📥 Customer Message Input</div>', unsafe_allow_html=True)
        
        customer_msg_input = st.text_area(
            "Customer Message",
            height=90,
            placeholder="Type or paste customer message here...",
            key="customer_input_field",
            label_visibility="collapsed"
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            process_btn = st.button("🚀 Process Customer Message", type="primary", use_container_width=True)
        with c2:
            st.button("🗑️ Clear Inputs", use_container_width=True, on_click=clear_inputs_callback)

        if process_btn:
            if not customer_msg_input.strip():
                st.warning("Please enter a customer message first.")
            elif st.session_state.get("session") is None:
                st.error("Backend session is not available.")
            else:
                with st.spinner("Analyzing message with AI models..."):
                    try:
                        analysis_result = st.session_state.session.on_customer_message(customer_msg_input.strip())
                        st.session_state.last_analysis = analysis_result
                        st.session_state.analyses_history.append(analysis_result)

                        if analysis_result.get("suggested_reply"):
                            st.session_state.suggested_reply = analysis_result["suggested_reply"]
                        else:
                            st.session_state.suggested_reply = ""
                        
                        st.success("Customer message analyzed successfully!")
                    except Exception as ex:
                        st.error(f"Error analyzing customer message: {ex}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel: Agent Response Input
        st.markdown('<div class="panel-card"><div class="panel-title">🎧 Agent Response & Evaluation</div>', unsafe_allow_html=True)
        
        agent_reply_text = st.text_area(
            "Agent Response Input",
            height=120,
            placeholder="Draft your reply to the customer here...",
            key="agent_response_field",
            label_visibility="collapsed"
        )

        b1, b2 = st.columns([1.2, 1])
        with b1:
            st.button("⭐ Send & Evaluate Response", type="primary", use_container_width=True, on_click=evaluate_response_callback)
        with b2:
            gen_suggest_btn = st.button("✨ Generate AI Reply", use_container_width=True)

        if gen_suggest_btn:
            if not st.session_state.get("session") or not st.session_state.session.last_customer_message:
                st.warning("Please process a customer message first before generating a suggestion.")
            else:
                with st.spinner("Generating AI suggested reply..."):
                    try:
                        reply = st.session_state.session.coach.suggest_reply(
                            customer_message=st.session_state.session.last_customer_message,
                            conversation_history=st.session_state.session.state.history
                        )
                        st.session_state.suggested_reply = reply
                        st.success("AI reply generated!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Could not generate AI reply: {ex}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Panel: Conversation History
        st.markdown('<div class="panel-card"><div class="panel-title">📜 Conversation History</div>', unsafe_allow_html=True)
        if st.session_state.get("session") and st.session_state.session.state.history:
            for idx, msg in enumerate(st.session_state.session.state.history):
                is_cust = msg.speaker.lower() == "customer"
                role_cls = "chat-customer" if is_cust else "chat-agent"
                speaker_label = "👤 Customer" if is_cust else "🎧 Support Agent"
                st.markdown(f'''
                <div class="chat-msg {role_cls}">
                    <div class="chat-meta">#{idx + 1} {speaker_label}</div>
                    {msg.text}
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.caption("No messages recorded in conversation history yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ai:
        # Panel: Real-time Analysis
        st.markdown('<div class="panel-card"><div class="panel-title">⚡ Real-Time AI Analysis</div>', unsafe_allow_html=True)
        if st.session_state.last_analysis:
            an = st.session_state.last_analysis

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Sentiment:**", unsafe_allow_html=True)
                st.markdown(render_badge(an.get("sentiment", "N/A"), "sentiment"), unsafe_allow_html=True)
                st.write("")
                st.markdown("**Urgency:**", unsafe_allow_html=True)
                st.markdown(render_badge(an.get("urgency", "N/A"), "urgency"), unsafe_allow_html=True)

            with rc2:
                st.markdown("**Escalation Risk:**", unsafe_allow_html=True)
                st.markdown(render_badge(an.get("escalation_risk", "N/A"), "escalation"), unsafe_allow_html=True)
                st.write("")
                st.markdown("**Detected Intent:**", unsafe_allow_html=True)
                intent_score_pct = round(an.get("intent_score", 0) * 100, 1)
                st.markdown(f'<span class="badge badge-blue">{an.get("intent", "N/A").upper()} ({intent_score_pct}%)</span>', unsafe_allow_html=True)

            st.divider()
            st.markdown("**Key Customer Issue:**")
            st.info(an.get("key_issue", "N/A"))
        else:
            st.caption("Awaiting customer message input to analyze sentiment, urgency, escalation risk, and intent.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel: AI Suggested Reply
        st.markdown('<div class="panel-card"><div class="panel-title">🤖 AI Suggested Reply</div>', unsafe_allow_html=True)
        if st.session_state.suggested_reply:
            st.write(st.session_state.suggested_reply)
            st.write("")
            st.button("📥 Use Suggestion (Copy to Agent Input)", use_container_width=True, on_click=use_suggestion_callback)
        else:
            st.caption("Suggested reply will automatically generate when escalation risk is HIGH, or you can click 'Generate AI Reply'.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Panel: Agent Coaching Feedback Summary
        st.markdown('<div class="panel-card"><div class="panel-title">🎯 Latest Response Coaching</div>', unsafe_allow_html=True)
        if st.session_state.last_feedback:
            fb = st.session_state.last_feedback
            
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Tone", f"{fb['tone_score']}/10")
            with sc2:
                st.metric("Empathy", f"{fb['empathy_score']}/10")
            with sc3:
                st.metric("Clarity", f"{fb['clarity_score']}/10")

            st.write("")
            st.markdown("**Coaching Tip:**")
            st.markdown(f'<div class="coaching-tip-card">💡 {fb["coaching_tip"]}</div>', unsafe_allow_html=True)
        else:
            st.caption("Submit an agent response to evaluate tone, empathy, clarity, and coaching tips.")
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 3: AI COACHING HUB
# -----------------------------------------------------------------------------
elif nav_choice == "🎯 AI Coaching":
    st.markdown('<div class="main-header">🎯 Agent AI Coaching Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed breakdown of communication quality, empathy scores, tone analysis, and action recommendations.</div>', unsafe_allow_html=True)

    if st.session_state.last_feedback:
        fb = st.session_state.last_feedback
        
        st.markdown('<div class="panel-card"><div class="panel-title">⭐ Latest Evaluation Breakdown</div>', unsafe_allow_html=True)
        
        c_tone, c_emp, c_clar = st.columns(3)

        with c_tone:
            st.subheader("Tone Score")
            st.progress(fb["tone_score"] / 10.0)
            st.markdown(f"**{fb['tone_score']} / 10**")

        with c_emp:
            st.subheader("Empathy Score")
            st.progress(fb["empathy_score"] / 10.0)
            st.markdown(f"**{fb['empathy_score']} / 10**")

        with c_clar:
            st.subheader("Clarity Score")
            st.progress(fb["clarity_score"] / 10.0)
            st.markdown(f"**{fb['clarity_score']} / 10**")

        st.divider()
        st.markdown("### 💡 Recommended Action Tip")
        st.markdown(f'<div class="coaching-tip-card" style="font-size:1.1rem;">{fb["coaching_tip"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("No response evaluated yet in this session. Send an agent response in the Conversation tab to see coaching feedback.")

    # Feedback history table
    if st.session_state.feedback_history:
        st.markdown('<div class="panel-card"><div class="panel-title">📜 Session Evaluation History</div>', unsafe_allow_html=True)
        df_fb = pd.DataFrame(st.session_state.feedback_history)
        df_fb.index = [f"Turn #{i+1}" for i in range(len(df_fb))]
        df_fb["Overall Avg"] = df_fb[["tone_score", "empathy_score", "clarity_score"]].mean(axis=1).round(1)
        st.dataframe(df_fb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# VIEW 4: ANALYTICS
# -----------------------------------------------------------------------------
elif nav_choice == "📈 Analytics":
    st.markdown('<div class="main-header">📈 Performance Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Session metrics, sentiment breakdown, escalation monitoring, and score distributions.</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    
    total_analyses = len(st.session_state.analyses_history)
    total_evaluations = len(st.session_state.feedback_history)
    high_risks = sum(1 for a in st.session_state.analyses_history if a.get("escalation_risk", "").lower() == "high")

    with a1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_analyses}</div><div class="metric-lbl">Customer Messages Analyzed</div></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{total_evaluations}</div><div class="metric-lbl">Agent Responses Evaluated</div></div>', unsafe_allow_html=True)
    with a3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{high_risks}</div><div class="metric-lbl">High Risk Escalations</div></div>', unsafe_allow_html=True)

    st.write("")
    st.write("")

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.markdown('<div class="panel-card"><div class="panel-title">📊 Sentiment Distribution</div>', unsafe_allow_html=True)
        if st.session_state.analyses_history:
            sentiments = [a.get("sentiment", "unknown").capitalize() for a in st.session_state.analyses_history]
            s_counts = pd.Series(sentiments).value_counts().reset_index()
            s_counts.columns = ["Sentiment", "Count"]
            st.bar_chart(s_counts, x="Sentiment", y="Count")
        else:
            st.caption("No sentiment data available for this session.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c_chart2:
        st.markdown('<div class="panel-card"><div class="panel-title">🎯 Coaching Scores Trend</div>', unsafe_allow_html=True)
        if st.session_state.feedback_history:
            df_chart = pd.DataFrame(st.session_state.feedback_history)[["tone_score", "empathy_score", "clarity_score"]]
            st.line_chart(df_chart)
        else:
            st.caption("No coaching evaluations available yet.")
        st.markdown('</div>', unsafe_allow_html=True)
