from sqlalchemy.ext.compiler import compiles
from sqlalchemy.schema import CreateTable
from sqlalchemy import *
from flasgger.utils import swag_from
from sqlalchemy import PrimaryKeyConstraint, Index
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, jsonify, request, render_template
import pandas as pd
from dateutil.parser import parse
from datetime import date, datetime
import sys
import csv
import os
import datetime
import pymysql

pymysql.install_as_MySQLdb()


app = Flask(__name__)


@app.errorhandler(Exception)
def internal_error(exception):
    app.logger.error(exception)
    data = {'error': 'Bad Request'}
    return jsonify(data), 500


@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route("/predict/<str:text>", methods=['GET'])
def predict(user_id):
    return jsonify({'user_id': user_id, 'error':
                    {'message': 'Not enough data to make prediction for this patient'}}), 404


@app.route("/health", methods=['GET'])
def health():
    return jsonify({}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
