from flask import Flask, jsonify, request, render_template


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
