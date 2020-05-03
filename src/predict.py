from flask import Flask, jsonify, request, render_template
import random
import pandas as pd

app = Flask(__name__)


@app.errorhandler(Exception)
def internal_error(exception):
    app.logger.error(exception)
    data = {'error': 'Bad Request'}
    return jsonify(data), 500


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/predict/<string:text>", methods=['GET'])
def predict(text):
    # temp

    diseases = [r'Диабет', r'Коронарус', r'Геморрой', r'ОРВИ', 
            r'Рак простаты', r'Трещина прямой кишки', r'Волчанка', 
            r'Порок сердца', r'Язва желудка', r'Болезнь Паркинсона']

    predictions = pd.DataFrame({r'Болезнь' : [str(r) for r in random.sample(diseases, 3)],
                                r'Вероятность' : sorted([random.random() for i in range(3)])[::-1]
                                }).to_json(orient='records',
                                        force_ascii=False)

    
    return jsonify(predictions), 200


@app.route("/health", methods=['GET'])
def health():
    return jsonify({}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
