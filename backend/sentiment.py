from transformers import pipeline


# Sentiment Analysis Model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment(customer_message):
    result = sentiment_analyzer(customer_message)[0]

    return {
        "label": result["label"],
        "score": result["score"]
    }