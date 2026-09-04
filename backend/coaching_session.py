from dataclasses import dataclass, field
from typing import List

from backend.ai_coach import AICoach, Message
from backend.intent import classify_intent


@dataclass
class ConversationState:
    """Stores the complete conversation and current customer risk."""

    history: List[Message] = field(default_factory=list)

    sentiment: str = "unknown"
    urgency: str = "unknown"
    escalation_risk: str = "unknown"
    key_issue: str = ""

    def add_message(self, speaker: str, text: str):

        self.history.append(
            Message(
                speaker=speaker,
                text=text
            )
        )


class RealTimeCoachingSession:

    def __init__(self):

        self.state = ConversationState()

        self.coach = AICoach()

        self.last_customer_message = ""

    # -------------------------
    # CUSTOMER MESSAGE
    # -------------------------

    def on_customer_message(
        self,
        message: str
    ):

        self.state.add_message(
            "customer",
            message
        )

        self.last_customer_message = message

        # Gemini customer analysis
        analysis = self.coach.analyze_customer_message(
            message
        )

        self.state.sentiment = analysis["sentiment"]

        self.state.urgency = analysis["urgency"]

        self.state.escalation_risk = analysis[
            "escalation_risk"
        ]

        self.state.key_issue = analysis[
            "key_issue"
        ]

        # Intent classification
        intent_result = classify_intent(message)

        suggested_reply = None

        # Generate reply when escalation risk is high
        if self.state.escalation_risk.lower() == "high":

            suggested_reply = self.coach.suggest_reply(
                customer_message=message,
                conversation_history=self.state.history
            )

        return {
            "sentiment": self.state.sentiment,
            "urgency": self.state.urgency,
            "escalation_risk": self.state.escalation_risk,
            "key_issue": self.state.key_issue,
            "intent": intent_result["intent"],
            "intent_score": intent_result["score"],
            "suggested_reply": suggested_reply
        }

    # -------------------------
    # AGENT MESSAGE
    # -------------------------

    def on_agent_message(
        self,
        message: str
    ):

        self.state.add_message(
            "agent",
            message
        )

        feedback = self.coach.evaluate_agent_response(
            customer_message=self.last_customer_message,
            agent_message=message
        )

        return {
            "tone_score": feedback.tone_score,
            "empathy_score": feedback.empathy_score,
            "clarity_score": feedback.clarity_score,
            "coaching_tip": feedback.coaching_tip
        }