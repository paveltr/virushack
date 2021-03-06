#! /usr/bin/env python
# -*- coding: utf-8 -*-

from google.cloud import storage
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
import youtokentome as yttm
import pickle
from google.cloud.storage import Client
from google.oauth2.service_account import Credentials
import os

warnings.filterwarnings('ignore')


def download_blob(storage_client, bucket_name,
                  source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(filename=destination_file_name,
                              client=storage_client)


def text_normalize(x, method='lemma'):
    if method == 'simple':
        x = ' '.join(r for r in re.findall(
            r'[а-я]+', str(x).lower()) if len(r) > 2)
        return x
    elif method == 'lemma':
        x = mystem.lemmatize(str(x).lower())
        x = [i for i in x if i != ' ' and i != '\n']
        x = ' '.join(x)
        x = re.sub(' +', ' ', x)
        return x


def remove_special_symbols(string):
    return re.sub('[^\w]+', ' ', string)


def replace_numbers(string):
    return re.sub(r'\b([\d]*)[\!\?\.,-]*([\d]+)\b', '<NUM>', string)


def tokenize(text):
    tokens = bpe.encode([text], output_type=yttm.OutputType.SUBWORD)
    return ' '.join(tokens[0])


def preprocess_text(text):
    text = text.lower()
    text = remove_special_symbols(text)
    text = replace_numbers(text)
    text = tokenize(text)
    return text


def make_prediction(text):
    text = preprocess_text(text)
    prediction = model_svm.predict_proba([text])[0].argsort()[::-1]
    probabilities = sorted(model_svm.predict_proba([text])[0], reverse=True)
    prediction = labelEncoder.inverse_transform(prediction)
    return {k: v for k, v in zip(prediction, probabilities)}


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

    pcp_dict = train.set_index('diagnos_cleaned')['doctor'].to_dict()
    icd_dict = train.set_index('diagnos_cleaned')['Код_диагноза'].to_dict()
    train = train[train['symptomps'].str.len() > 0]

    diag_freq = train['diagnos_cleaned'].value_counts()
    train = train[train['diagnos_cleaned'].isin(
        diag_freq[diag_freq >= 20].index.tolist())]

    # Pipeline

    y = train['diagnos_cleaned'].values

    clf = CalibratedClassifierCV(
        base_estimator=svm.LinearSVC(C=.1, class_weight='balanced'),
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

    return model, pcp_dict, icd_dict


warnings.filterwarnings('ignore')


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

global clf, pcp_dict, mystem, icd_dict, model_svm, labelEncoder
clf, pcp_dict, icd_dict = get_model()
mystem = Mystem()

client = Client(project='cosmic-rarity-277009',
                credentials=Credentials.from_service_account_file(os.environ['MASTER_JSON']))

BPE_PATH = '/var/www/src/models/bpe.model'
LENCODER_PATH = '/var/www/src/models/labelEncoder.pickle'
MODEL_PATH = '/var/www/src/models/model_svm.pickle'

download_blob(client, 'mymed-models',
              'doctor-prediction/20200729/%s' % 'bpe.model', BPE_PATH)
download_blob(client, 'mymed-models',
              'doctor-prediction/20200729/%s' % 'labelEncoder.pickle', LENCODER_PATH)
download_blob(client, 'mymed-models',
              'doctor-prediction/20200729/%s' % 'model_svm.pickle', MODEL_PATH)

bpe = yttm.BPE(model=BPE_PATH)
with open(MODEL_PATH, 'rb') as handle:
    model_svm = pickle.load(handle)

with open(LENCODER_PATH, 'rb') as handle:
    labelEncoder = pickle.load(handle)


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


def check_text(text):
    text = str(text)
    if len(text) < 3:
        return r'error'
    else:
        return text


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


@app.route("/predict_doctor", methods=['GET', 'POST'])
def predict_doctor():
    if request.method == 'POST':
        text = check_text(request.json['text'])
    elif request.method == 'GET':
        text = check_text(request.args.get('text', r'error'))

    if text == r'error':
        return jsonify({'prediction': 'bad input data'}), 200
    else:
        return jsonify({'prediction': str(make_prediction(text))}), 200


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
                               'ICD': ['Уточните симптомы, недостаточно данных'],
                               'Вероятность2': [-1],
                               'Болезнь2': ['Уточните симптомы, недостаточно данных'],
                               'Доктор2': ['Уточните симптомы, недостаточно данных'],
                               'Диагноз2': ['Уточните симптомы, недостаточно данных'],
                               'ICD': ['Уточните симптомы, недостаточно данных'],
                               'Вероятность3': [-1],
                               'Болезнь3': ['Уточните симптомы, недостаточно данных'],
                               'Доктор3': ['Уточните симптомы, недостаточно данных'],
                               'Диагноз3': ['Уточните симптомы, недостаточно данных'],
                               'ICD': ['Уточните симптомы, недостаточно данных']
                               })
        return jsonify(result.to_json(orient='records', force_ascii=False)), 200

    data = pd.DataFrame({'symptomps': [symptomps],
                         'age': [age],
                         'gender': [gender]})

    prediction = pd.DataFrame(predict_diag(clf, data)[0], columns=['Вероятность',
                                                                   'Болезнь'])
    prediction['Доктор'] = prediction['Болезнь'].map(pcp_dict)
    prediction['Диагноз'] = prediction['Болезнь'].map(parse_diag)
    prediction['ICD'] = prediction['Болезнь'].map(icd_dict)

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
