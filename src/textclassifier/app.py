from flask import Flask, request
from transformers import RobertaTokenizerFast, \
    TFRobertaForSequenceClassification, pipeline


app = Flask(__name__)
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion_classifier = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
sentiment_classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")


sentiments = ("negative", "positive")


@app.post('/classify')
def classify_text():
    body = request.json
    data = [row['text'] for row in body['conversation']]

    return {
        'emotions': emotion_classifier(data),
        'overall_sentiment': sentiment_classifier([
            body['transcript']
        ])[0]
    }


if __name__ == '__main__':
    app.run(port=5000)
