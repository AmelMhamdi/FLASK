from flask import Flask, request, jsonify
from flasgger import Swagger
import pickle

import numpy as np

with open('C:/Users/Amel/Documents/IAworkspace/FLASK/FLASK/rf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/api')
def predict_iris():
    """Example endpoint returning a prediction of iris
    This is using docstrings for specifications
           ---
           parameters:
               - name: s_length
                 in: query
                 type: number
                 required: true
               - name: s_width
                 in: query
                 type: number
                 required: true
               - name: p_length
                 in: query
                 type: number
                 required: true
               - name: p_width
                 in: query
                 type: number
                 required: true
          """
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")

    prediction = model.predict(np.array([[s_length, s_width, p_length, p_width]]))

    return str(prediction)


if __name__ == '__main__':
    app.run()
