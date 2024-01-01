from flask import Flask, render_template, redirect,request
import pickle, pandas as pd, numpy as np, matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
app = Flask(__name__)

@app.route("/", methods=['POST',"GET"])
def Home():
    price = 0
    if request.method=='POST':
        features = {
            'area' : int(request.form['area']),
            'stories' : int(request.form['stories']),
            'basement' : int(request.form['basement']),
            'hotwaterheating' : int(request.form['hotwaterheating']),
            'airconditioning' : int(request.form['airconditioning']),
            'parking' : int(request.form['parking']),
            'prefarea' : int(request.form['prefarea']),
            'unfurnished' : int(request.form['unfurnished']),
            'bathroom_per_bed' : int(request.form['bathroom'])/int(request.form['bedroom']),
            }
        pik_file = pickle.load(open('house.pkl','rb'))
        rescale_col = ['area', 'bathroom_per_bed', 'stories', 'parking']

        scale_price = pik_file['scale_price']
        scaler = pik_file['scale']
        model = pik_file['Lmodel']

        df = pd.DataFrame([features])
        df[rescale_col] = scaler.transform(df[rescale_col])
        df = np.array(df)
        price = model.predict(df).reshape(-1,1)
        price = round(scale_price.inverse_transform(price)[0][0],2)
        print(price)

    return render_template('index.html', price = price)

if __name__ == "__main__":
    app.run(debug=True)