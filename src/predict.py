from flask import Flask, jsonify, request, render_template
import random
import pandas as pd
from lib.classifier import *
from joblib import load

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

clf = load('/var/www/src/model/mymed_v0.joblib')
pcp_dict = load('/var/www/src/model/pcp_dict.joblib')


def check_age(age):
    try:
        if 0 < int(age) < 200:
            return str(age)
        else:
            return str(35)
    except ValueError:
        return str(35)


def check_gender(gender):
    if gender not in [r'мужской', r'женский']:
        return r'женский'
    return str(gender)


def check_diag(diag):
    diag = str(diag)
    if len(diag) < 3:
        return r'Ничего'
    else:
        return text_normalize(diag)


@app.errorhandler(Exception)
def internal_error(exception):
    app.logger.error(exception)
    data = {'error': 'Bad Request'}
    return jsonify(data), 500


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict(text):

    if request.method == 'POST':
        symptomps = check_diag(request.json['symptomps'])
        age = check_age(request.json['age'])
        gender = check_gender(request.json['gender'])

    elif request.method == 'GET':
        symptomps = check_diag(request.args.get('symptomps', r'Ничего'))
        age = check_age(request.args.get('age', r'35'))
        gender = check_gender(request.args.get('gender', r'женский'))

    data = pd.DataFrame({'symptomps': [symptomps],
                         'age': [age],
                         'gender': [gender]})

    prediction = pd.DataFrame(predict_diag(clf, data)[0], columns=['Вероятность', 
                                                                   'Болезнь'])
    prediction['Доктор'] = prediction['Болезнь'].map(pcp_dict)

    return jsonify(prediction.to_json(orient='records', force_ascii=False)), 200


@app.route("/health", methods=['GET'])
def health():
    return jsonify({}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
