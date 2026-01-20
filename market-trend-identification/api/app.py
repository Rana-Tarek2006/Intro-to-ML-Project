# Flask API for predictions
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return jsonify({'status': 'API endpoint'})

if __name__ == '__main__':
    app.run(debug=True)
