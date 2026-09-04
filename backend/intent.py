from transformers import pipeline


# Intent Classification Model
intent_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


# Intent categories from your original code
INTENTS = [
    "complaint",
    "query",
    "purchase",
    "technical issue",
    "feedback"
]


def classify_intent(customer_message):
    result = intent_classifier(
        customer_message,
        candidate_labels=INTENTS
    )

    return {
        "intent": result["labels"][0],
        "score": result["scores"][0]
    }