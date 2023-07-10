from flask import Flask, request
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline


app = Flask(__name__)
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
emotion_classifier = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    

@app.post('/classify')
def classify_text():
    body = request.json
    data = [row['text'] for row in  body['inputs']]
    emotion_labels = emotion_classifier(data)
    return {
        'data': emotion_labels,
        'status': 'success'
    }


@app.get('/test/<int:value>')
def test_endpoint(value):
    return str(value)


if __name__ == '__main__':
    app.run(port=500)
