from flask import Flask, request
from transformers import RobertaTokenizerFast, \
    TFRobertaForSequenceClassification, pipeline


app = Flask(__name__)
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion_classifier = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


sentiments = ("negative", "positive")


@app.post('/classify')
def classify_text():
    body = request.json
    result = {}
    messages = body.get('messages')
    transcript = body.get('transcript')

    if messages:
        result['emotions'] = emotion_classifier(messages)

    if transcript:
        result['overall_sentiment'] = sentiment_classifier(transcript)

    return result


if __name__ == '__main__':
    app.run(port=8000)
