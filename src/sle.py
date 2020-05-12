#! /usr/bin/env python
# -*- coding: utf-8 -*-

from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from flask import Flask, jsonify, request, render_template
import random
import pandas as pd
import re
import warnings
from pymystem3 import Mystem

warnings.filterwarnings('ignore')


def text_normalize(x, method='lemma'):
    x = ' '.join(r for r in re.findall(r'[а-я]+', str(x).lower()) if len(r) > 2)
    if method == 'simple':
        return x
    elif method == 'lemma':
        x = mystem.lemmatize(x)
        x = [i for i in x if i != ' ' and i != '\n']
        x = ' '.join(x)
        x = re.sub(' +', ' ', x)
        return x


def top_pair(values, keys, n=3):
    return sorted(zip(values, keys), reverse=True)[:n]


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        if self.key == 'symptomps':
            return data_dict[self.key]
        else:
            return data_dict[self.key].values.reshape(-1, 1)


def get_model():

    # Data

    train = pd.read_csv(r'/var/www/src/train_data.csv.tar.gz')
    train['symptomps'] = train['galobi_lem'].values
    train['gender'] = train['Пол'].map(
        lambda x: 'мужской' if x == 1 else 'женский')
    train['age'] = train['Возраст'].astype(str).values
    
    pcp_dict = train.set_index('Диагноз')['doctor'].to_dict()
    train = train[train['symptomps'].str.len() > 0]

    diag_freq = train['Диагноз'].value_counts()
    train = train[train['Диагноз'].isin(diag_freq[diag_freq >= 20].index.tolist())]
    
    # Pipeline

    y = train['Диагноз'].values

    clf = CalibratedClassifierCV(
        base_estimator=BaggingClassifier(svm.LinearSVC(C=.1, class_weight='balanced'),
                                         n_estimators=5, max_samples=.2,
                                         bootstrap=False),
        method='isotonic',
        cv=3)

    model = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[

                # Pipeline for pulling features
                    ('tfidf', Pipeline([
                        ('selector', ItemSelector(key='symptomps')),
                        ('tdidf', TfidfVectorizer(
                            analyzer='char', ngram_range=(1, 5)))
                    ])),
                ('age', Pipeline([
                    ('selector', ItemSelector(key='age')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
                ('gender', Pipeline([
                    ('selector', ItemSelector(key='gender')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore'))
                ])),
            ],

            # weight components in FeatureUnion
            transformer_weights={
                'tfidf': 0.8,
                'age': 0.1,
                'gender': 0.1
            },
        )),
        ('svc', clf)])

    model.fit(train, y)

    return model, pcp_dict


warnings.filterwarnings('ignore')


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

global clf, pcp_dict, mystem
clf, pcp_dict = get_model()
mystem = Mystem()


def predict_diag(model, X):
    predictions = model.predict_proba(X)
    classes = model.classes_
    return [top_pair(p, classes) for p in predictions]


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
        return r'error'
    else:
        return text_normalize(diag)


def parse_diag(x):
    return x.split(r'[')[0].split(r',')[0].split(r' и ')[0]


@app.errorhandler(Exception)
def internal_error(exception):
    app.logger.error(exception)
    data = {'error': 'Bad Request'}
    return jsonify(data), 500


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        symptomps = check_diag(request.json['symptomps'])
        age = check_age(request.json['age'])
        gender = check_gender(request.json['gender'])

    elif request.method == 'GET':
        symptomps = check_diag(request.args.get('symptomps', r'error'))
        age = check_age(request.args.get('age', r'35'))
        gender = check_gender(request.args.get('gender', r'женский'))

    if symptomps == r'error':
        result = pd.DataFrame({'Вероятность1': [-1],
                               'Болезнь1': ['Уточните симптомы, недостаточно данных'],
                               'Доктор1': ['Уточните симптомы, недостаточно данных'],
                               'Диагноз1': ['Уточните симптомы, недостаточно данных'],
                               'Вероятность2': [-1],
                               'Болезнь2': ['Уточните симптомы, недостаточно данных'],
                               'Доктор2': ['Уточните симптомы, недостаточно данных'],
                               'Диагноз2': ['Уточните симптомы, недостаточно данных'],
                               'Вероятность3': [-1],
                               'Болезнь3': ['Уточните симптомы, недостаточно данных'],
                               'Доктор3': ['Уточните симптомы, недостаточно данных'],
                               'Диагноз3': ['Уточните симптомы, недостаточно данных']})
        return jsonify(result.to_json(orient='records', force_ascii=False)), 200

    data = pd.DataFrame({'symptomps': [symptomps],
                         'age': [age],
                         'gender': [gender]})

    prediction = pd.DataFrame(predict_diag(clf, data)[0], columns=['Вероятность',
                                                                   'Болезнь'])
    prediction['Доктор'] = prediction['Болезнь'].map(pcp_dict)
    prediction['Диагноз'] = prediction['Болезнь'].map(parse_diag)

    result = pd.DataFrame({'key': [1]})
    for i in range(3):
        merge_df = prediction.iloc[i:i+1, :].rename(columns=dict(zip(prediction.columns,
                                                                     [c + str(i+1)
                                                                      for c in prediction.columns])))
        merge_df['key'] = 1
        result = result.merge(merge_df, how='left', on='key')
    result.drop('key', axis=1, inplace=True)
    return jsonify(result.to_json(orient='records', force_ascii=False)), 200


@app.route("/health", methods=['GET'])
def health():
    return jsonify({}), 200


if __name__ == '__main__':
    app.run(debug=True)
