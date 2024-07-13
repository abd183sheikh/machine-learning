import pandas as pd
from flask import Flask, render_template, request, redirect
import numpy as np
# from flask_cors import CORS,cross_origin
import pickle
app = Flask(__name__)
car = pd.read_csv("cleaned_car.csv")
# cors=CORS(app)
model = pickle.load(open('car_prediction_model.pkl', 'rb'))
# print(car["company"].unique())


@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_model = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = sorted(car["fuel_type"].unique())
    return render_template("index.html", companies=companies, car_model=car_model, years=year, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():

    company = request.form.get('company')
    car_model = request.form.get('model')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel')
    driven = request.form.get('kilometer')

    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == "__main__":
    app.run(debug=True)
