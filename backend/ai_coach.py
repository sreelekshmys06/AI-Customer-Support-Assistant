import os
import json
import time
from dataclasses import dataclass
from typing import List, Optional

from google import genai

from dotenv import load_dotenv

load_dotenv()

@dataclass
class Message:
    """Represents one message in the conversation."""
    speaker: str
    text: str


@dataclass
class CoachingFeedback:
    """Stores feedback about the agent's response."""
    tone_score: int
    empathy_score: int
    clarity_score: int
    coaching_tip: str


class AICoach:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.6-flash"

    def _parse_json(self, text: str) -> dict:

        text = text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

        try:
            result = json.loads(text)

        except json.JSONDecodeError as e:

            print("\nCould not parse Gemini response as JSON.")
            print("Raw response:")
            print(text)

            raise ValueError(
                f"Invalid JSON returned by Gemini: {e}"
            )

        return result

    def _generate_content(self, prompt: str):

        for attempt in range(3):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )

                return response

            except Exception as e:

                error_message = str(e)

                if (
                    "503" in error_message
                    or "UNAVAILABLE" in error_message
                ):

                    if attempt < 2:

                        print(
                            f"\nGemini server is temporarily busy."
                            f" Retrying... ({attempt + 1}/3)"
                        )

                        time.sleep(3)

                    else:

                        raise RuntimeError(
                            "Gemini is temporarily unavailable. "
                            "Please try again after a few minutes."
                        )

                else:
                    raise e

    def analyze_customer_message(
        self,
        customer_message: str
    ) -> dict:

        prompt = f"""
You are an AI customer-support risk analyzer.

Analyze the following customer message.

Customer message:
{customer_message}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "sentiment": "positive|neutral|negative",
    "urgency": "low|medium|high",
    "escalation_risk": "low|medium|high",
    "key_issue": "short description of the main issue"
}}

Do not add explanations outside the JSON.
"""

        response = self._generate_content(prompt)

        text = response.text

        return self._parse_json(text)

    def evaluate_agent_response(
        self,
        customer_message: str,
        agent_message: str
    ) -> CoachingFeedback:

        prompt = f"""
You are an AI customer-support coach.

Evaluate the agent's response to the customer.

Customer message:
{customer_message}

Agent response:
{agent_message}

Score the agent from 1 to 10 for:

1. Tone
2. Empathy
3. Clarity

Then provide ONE concrete coaching tip.

Return ONLY valid JSON using exactly this structure:

{{
    "tone_score": 1,
    "empathy_score": 1,
    "clarity_score": 1,
    "coaching_tip": "one concrete coaching tip"
}}

Do not add any explanation outside the JSON.
"""

        response = self._generate_content(prompt)

        text = response.text

        result = self._parse_json(text)

        return CoachingFeedback(
            tone_score=int(result["tone_score"]),
            empathy_score=int(result["empathy_score"]),
            clarity_score=int(result["clarity_score"]),
            coaching_tip=result["coaching_tip"]
        )

    def suggest_reply(
        self,
        customer_message: str,
        conversation_history: Optional[List[Message]] = None
    ) -> str:

        history_text = ""

        if conversation_history:

            history_text = "\n".join(
                f"{message.speaker}: {message.text}"
                for message in conversation_history
            )

        prompt = f"""
You are an expert customer-support agent.

Create a professional, empathetic and concise reply to the customer.

Customer message:
{customer_message}

Previous conversation:
{history_text}

The customer may be frustrated.

Requirements:

- Acknowledge the customer's concern.
- Show empathy.
- Clearly explain the next step if possible.
- Do not make unsupported promises.
- Keep the response concise.
- Return only the reply text.
"""

        response = self._generate_content(prompt)

        return response.text.strip()