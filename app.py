from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn import tree
import pickle

app = Flask(__name__)

model_file = open('model.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    return render_template('index.html', prediction=0)

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    SMT1, SMT2, SMT3, SMT4, SMT5 = [x for x in request.form.values()]

    data = []

    data.append(int(SMT1))
    data.append(int(SMT2))
    data.append(int(SMT3))
    data.append(int(SMT4))
    data.append(int(SMT5))    
    
    
    prediction = model.predict([data])
    output = round(prediction[0], 2)

    return render_template('index.html', prediction=output, SMT1=SMT1, SMT2=SMT2, SMT3=SMT3, SMT4=SMT4, SMT5=SMT5)


if __name__ == '__main__':
    app.run(debug=True)