from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request, render_template
import random
import pandas as pd
from joblib import load
import re
import warnings
import pickle

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def text_normalize(x):
    return ' '.join(r for r in re.findall(r'[а-я]+', str(x).lower())
                    if len(r) > 2)


def top_pair(values, keys, n=3):
    return sorted(zip(values, keys), reverse=True)[:n]


def predict_diag(model, X):
    predictions = model.predict_proba(X)
    classes = model.classes_
    return [top_pair(p, classes) for p in predictions]


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


if __name__ == '__main__':

    global clf, pcp_dict

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
    def predict():

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

    app.run(host='0.0.0.0', port=8000)
