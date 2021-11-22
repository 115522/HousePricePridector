import flask
from flask import Flask , render_template,request
import numpy as np
import joblib
import pandas as pd


data_preparation = joblib.load(open('data_prep.pkl','rb')) #charger le full pipeline
Best_model = joblib.load(open('clf__GradientBoostingRegressor.pkl','rb'))  #charger le modéle 


app = Flask(__name__)

@app.route('/')

def man():
    return render_template('index.html')
@app.route('/predict',methods =['POST'])
def home():
    longitude = request.form.get('longitude')
    latitude = request.form.get('latitude')
    housing_median_age = request.form.get('housing_median_age')
    total_rooms = request.form.get('total_rooms')
    total_bedrooms = request.form.get('total_bedrooms')
    population = request.form.get('population')
    households = request.form.get('households')
    median_income = request.form.get('median_income')  
    ocean_proximity = request.form['option']

     #Feature Engineering : Combinaison d'attributs
    rooms_per_household = float(total_rooms)/ float(households)
    bedrooms_per_room = float(total_bedrooms)/ float(total_rooms)
    population_per_household = float(population)/ float(households)

    features = np.array([longitude,latitude,housing_median_age,total_rooms,population,households,median_income,ocean_proximity,rooms_per_household,bedrooms_per_room,population_per_household])
#convert from array to dataframe
    features_df = pd.DataFrame(data=[features],columns=['longitude','latitude','housing_median_age','total_rooms','population','households','median_income','ocean_proximity','rooms_per_household','bedrooms_per_room','population_per_household'])
    clean_features = data_preparation.transform(features_df)
    house_price_prediction = Best_model.predict(clean_features)

    return render_template("prediction.html",prediction_text='price of house chosen is:{}'.format(house_price_prediction))
if __name__ == '__main__':
    app.run(debug=True)