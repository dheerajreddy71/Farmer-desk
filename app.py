from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and prepare datasets for yield prediction
yield_df = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/blob/main/yield_df.csv")
crop_recommendation_data = pd.read_csv("D:/DESIGN PROJECT/Crop_recommendation.csv")

yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

# Load crop data and train the model for temperature prediction
data = pd.read_csv("https://github.com/dheerajreddy71/Design_Project/blob/main/ds1.csv", encoding='ISO-8859-1')
data = data.drop(['Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7'], axis=1)
X = data.drop(['Crop', 'Temperature Required (°F)'], axis=1)
y = data['Temperature Required (°F)']
model = LinearRegression()
model.fit(X, y)

# Function to predict temperature and humidity requirements for a crop
def predict_requirements(crop_name):
    crop_name = crop_name.lower()
    crop_data = data[data['Crop'].str.lower() == crop_name].drop(['Crop', 'Temperature Required (°F)'], axis=1)
    if crop_data.empty:
        return None, None  # Handle cases where crop_name is not found
    predicted_temperature = model.predict(crop_data)
    crop_row = data[data['Crop'].str.lower() == crop_name]
    humidity_required = crop_row['Humidity Required (%)'].values[0]
    return humidity_required, predicted_temperature[0]

# Function to get pest warnings for a crop
crop_pest_data = {}
planting_time_info = {}
growth_stage_info = {}
pesticides_info = {}

# Read data from the CSV file and store it in dictionaries
with open("https://github.com/dheerajreddy71/Design_Project/blob/main/ds2.csv", 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        if len(row) >= 2:
            crop = row[0].strip().lower()
            pest = row[1].strip()
            crop_pest_data[crop] = pest
            humidity_data = row[2].strip()
        if len(row) >= 3:
            crop = row[0].strip().lower()
            planting_time = row[5].strip()
            planting_time_info[crop] = planting_time
            growth_stage = row[6].strip()
            growth_stage_info[crop] = growth_stage
            pesticides_row = row[4].strip()
            pesticides_info[crop] = pesticides_row

def predict_pest_warnings(crop_name):
    crop_name = crop_name.lower()
    specified_crops = [crop_name]

    pest_warnings = []

    for crop in specified_crops:
        if crop in crop_pest_data:
            pests = crop_pest_data[crop].split(', ')
            warning_message = f"\nBeware of pests like {', '.join(pests)} for {crop.capitalize()}.\n"

            if crop in planting_time_info:
                planting_time = planting_time_info[crop]
                warning_message += f"\nPlanting Time: {planting_time}\n"

            if crop in growth_stage_info:
                growth_stage = growth_stage_info[crop]
                warning_message += f"\nGrowth Stages of Plant: {growth_stage}\n"

            if crop in pesticides_info:
                pesticides = pesticides_info[crop]
                warning_message += f"\nUse Pesticides like: {pesticides}\n"
                
            pest_warnings.append(warning_message)

    return '\n'.join(pest_warnings)

# Load and preprocess crop price data
price_data = pd.read_csv('https://github.com/dheerajreddy71/Design_Project/blob/main/pred_data.csv', encoding='ISO-8859-1')
price_data['arrival_date'] = pd.to_datetime(price_data['arrival_date'])
price_data['day'] = price_data['arrival_date'].dt.day
price_data['month'] = price_data['arrival_date'].dt.month
price_data['year'] = price_data['arrival_date'].dt.year
price_data.drop(['arrival_date'], axis=1, inplace=True)

price_X = price_data.drop(['min_price', 'max_price', 'modal_price'], axis=1)
price_y = price_data[['min_price', 'max_price', 'modal_price']]

price_encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['state', 'district', 'market', 'commodity', 'variety'])
    ],
    remainder='passthrough'
)

price_X_encoded = price_encoder.fit_transform(price_X)
price_X_train, price_X_test, price_y_train, price_y_test = train_test_split(price_X_encoded, price_y, test_size=0.2, random_state=42)

price_model = LinearRegression()
price_model.fit(price_X_train, price_y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    yield_result = None
    crop_result = None
    min_price = max_price = modal_price = None
    if request.method == 'POST':
        if 'predict_yield' in request.form:
            features = {
                'Year': int(request.form['year']),
                'average_rain_fall_mm_per_year': float(request.form['rainfall']),
                'pesticides_tonnes': float(request.form['pesticides']),
                'avg_temp': float(request.form['temp']),
                'Area': request.form['area'],
                'Item': request.form['item'],
            }
            features_array = np.array([[features['Year'], features['average_rain_fall_mm_per_year'],
                                        features['pesticides_tonnes'], features['avg_temp'],
                                        features['Area'], features['Item']]], dtype=object)
            transformed_features = yield_preprocessor.transform(features_array)
            predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)
            yield_result = f"The predicted yield is {predicted_yield[0][0]:.2f} hectograms (hg) per hectare (ha)."

        elif 'recommend_crop' in request.form:
            crop_features = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            recommended_crop = crop_model.predict([crop_features])[0]
            yield_result = f"Recommended Crop: {recommended_crop}"

        elif 'calculate' in request.form:
            crop_name = request.form.get('crop_name')
            if crop_name:
                humidity, temperature = predict_requirements(crop_name)
                pest_warning = predict_pest_warnings(crop_name)
                crop_result = {
                    'humidity': humidity,
                    'temperature': temperature,
                    'pest_warning': pest_warning
                }

        elif 'predict_prices' in request.form:
            state = request.form.get('state')
            district = request.form.get('district')
            market = request.form.get('market')
            commodity = request.form.get('commodity')
            variety = request.form.get('variety')
            arrival_date = request.form.get('arrival_date')

            if not all([state, district, market, commodity, variety, arrival_date]):
                return render_template('index.html', min_price=None, max_price=None, modal_price=None)

            input_data = {
                'state': state,
                'district': district,
                'market': market,
                'commodity': commodity,
                'variety': variety,
                'arrival_date': pd.to_datetime(arrival_date)
            }

            input_df = pd.DataFrame([input_data])
            input_df['day'] = input_df['arrival_date'].dt.day
            input_df['month'] = input_df['arrival_date'].dt.month
            input_df['year'] = input_df['arrival_date'].dt.year
            input_df.drop(['arrival_date'], axis=1, inplace=True)
            input_encoded = price_encoder.transform(input_df)

            predicted_prices = price_model.predict(input_encoded)
            min_price, max_price, modal_price = predicted_prices[0]

    return render_template('index.html', yield_result=yield_result, crop_result=crop_result, min_price=min_price, max_price=max_price, modal_price=modal_price)

if __name__ == '__main__':
    app.run(debug=True)
