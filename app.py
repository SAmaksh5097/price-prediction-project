from flask import Flask, render_template, request, url_for
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import os # <-- Import the os library

app = Flask(__name__)

# --- START: ROBUST FILE PATH LOGIC ---
# Get the absolute path of the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Define the full paths for your data and model files
DATA_FILE = os.path.join(basedir, 'dataset_enriched.xlsx')
MODEL_FILE = os.path.join(basedir, 'model', 'vehicle_model.pkl')
# --- END: ROBUST FILE PATH LOGIC ---


BIKE_COMPANIES = ['Honda', 'Royal Enfield', 'TVS', 'Ola Electric', 'Bajaj', 'Ather']

def clean_data(df):
    # (This function is unchanged)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kms_driven'] = df['kms_driven'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['year', 'kms_driven', 'Price'], inplace=True)
    df['year'] = df['year'].astype(int)
    df['kms_driven'] = df['kms_driven'].astype(int)
    price_threshold = 1000
    df['Price'] = df['Price'].apply(lambda x: x / 100000 if x > price_threshold else x)
    cat_features = ['company', 'fuel_type', 'city', 'seller_type', 'name']
    for col in cat_features:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()
    df.fillna({'battery_capacity_kwh': 0, 'range_km': 0, 'engine_cc': 0, 'has_sunroof': False, 'has_adas': False}, inplace=True)
    df['vehicle_type'] = np.where(df['company'].isin(BIKE_COMPANIES), 'Two-Wheeler', 'Car')
    df['age'] = datetime.now().year - df['year']
    return df

try:
    # Use the full path variables to load files
    raw_df = pd.read_excel(DATA_FILE)
    full_df = clean_data(raw_df.copy())
    print(f"✅ Enriched dataset loaded and cleaned. {len(full_df)} valid rows available.")
except Exception as e:
    full_df = None
    print(f"❌ FATAL ERROR loading or cleaning dataset: {e}")

try:
    # Use the full path variables to load files
    with open(MODEL_FILE, 'rb') as f:
        pipeline = pickle.load(f)
    print("✅ Final model pipeline loaded.")
except Exception as e:
    pipeline = None
    print(f"❌ FATAL ERROR loading model: {e}")

# ... (The rest of the file is exactly the same as before) ...
def get_form_data(vehicle_type):
    if full_df is None: return {}
    df = full_df[full_df['vehicle_type'] == vehicle_type]
    
    existing_years = list(df['year'].unique())
    additional_years = [2022, 2021, 2020, 2019]
    all_years = sorted(list(set(existing_years + additional_years)), reverse=True)

    data = {
        'companies': sorted([str(item) for item in df['company'].unique()]),
        'years': all_years,
        'cities': sorted([str(item) for item in df['city'].unique()])
    }
    if vehicle_type == 'Car':
        data['fuel_types'] = sorted([str(f) for f in df['fuel_type'].unique()])
    return data

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/cars')
def car_form():
    return render_template('car_form.html', **get_form_data('Car'))

@app.route('/bikes')
def bike_form():
    return render_template('bike_form.html', **get_form_data('Two-Wheeler'))

@app.route('/buy')
def buyer_dashboard():
    if full_df is None or pipeline is None: return "Server Error", 500
    
    brand = request.args.get('brand')
    max_price = request.args.get('max_price', type=float)
    max_age = request.args.get('max_age', type=int)

    filtered_df = full_df.copy()

    if brand and brand != "Any Brand": filtered_df = filtered_df[filtered_df['company'] == brand]
    if max_price: filtered_df = filtered_df[filtered_df['Price'] <= max_price]
    if max_age: filtered_df = filtered_df[filtered_df['age'] <= max_age]
        
    sample_size = min(15, len(filtered_df))
    if sample_size > 0:
        inventory_df = filtered_df.sample(n=sample_size, random_state=42).copy()
        inventory_df['predicted_price'] = [float(p) for p in pipeline.predict(inventory_df)]
        vehicles_list = inventory_df.to_dict(orient='records')
        
        for v in vehicles_list:
            if v['Price'] < v['predicted_price'] * 0.95: v.update({'deal_status': 'Great Deal!', 'deal_color': 'success'})
            elif v['Price'] > v['predicted_price'] * 1.05: v.update({'deal_status': 'Scope for Negotiation', 'deal_color': 'warning'})
            else: v.update({'deal_status': 'Fair Price', 'deal_color': 'primary'})
    else:
        vehicles_list = []

    all_companies = sorted(full_df['company'].unique())
    
    return render_template(
        'buyer_dashboard.html', 
        vehicles=vehicles_list, 
        all_companies=all_companies,
        current_filters={'brand': brand, 'max_price': max_price, 'max_age': max_age}
    )

@app.route('/dashboard')
def dashboard():
    if full_df is None: return "Server Error", 500
    avg_price_by_company = full_df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
    fuel_type_counts = full_df['fuel_type'].value_counts()
    city_counts = full_df['city'].value_counts().head(10)
    scatter_df = full_df.sample(n=min(len(full_df), 300), random_state=42)
    age_price_data = [list(rec) for rec in scatter_df[['age', 'Price']].to_records(index=False)]

    return render_template(
        'dashboard.html',
        company_labels=avg_price_by_company.index.tolist(),
        price_values=avg_price_by_company.values.tolist(),
        fuel_labels=fuel_type_counts.index.tolist(),
        fuel_values=fuel_type_counts.values.tolist(),
        city_labels=city_counts.index.tolist(),
        city_values=city_counts.values.tolist(),
        age_price_data=age_price_data
    )

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None: return "Model not loaded.", 500
    try:
        vehicle_type = request.form['vehicle_type']
        form_data = {'company': request.form['company'], 'year': int(request.form['year']), 'kms_driven': int(request.form['kms_driven']), 'city': request.form['city'], 'vehicle_type': vehicle_type, 'seller_type': 'Individual', 'range_km': 0, 'has_sunroof': False, 'has_adas': False, 'name': 'N/A'}
        seller_offer = float(request.form.get('offer_price', 0))
        form_data['offer_price'] = seller_offer

        if vehicle_type == 'Car':
            form_data.update({'fuel_type': request.form['fuel_type'], 'battery_capacity_kwh': float(request.form.get('battery_capacity_kwh', 0)), 'engine_cc': 0})
            template, dropdowns = 'car_form.html', get_form_data('Car')
        else:
            form_data.update({'fuel_type': 'Petrol', 'engine_cc': int(request.form.get('engine_cc', 0)), 'battery_capacity_kwh': 0})
            template, dropdowns = 'bike_form.html', get_form_data('Two-Wheeler')
        
        input_df = pd.DataFrame([form_data])
        input_df['age'] = datetime.now().year - input_df['year']
        
        current_prediction = float(pipeline.predict(input_df)[0])
        
        forecast_prices = []
        future_input_df = input_df.copy()
        for i in range(1, 4):
            future_input_df['age'] = input_df['age'] + i
            future_input_df['kms_driven'] = input_df['kms_driven'] + (i * 10000)
            future_pred = float(pipeline.predict(future_input_df)[0])
            forecast_prices.append(max(future_pred, 0.2))
        
        forecast_data = {"labels": ["Today", "+1 Year", "+2 Years", "+3 Years"], "values": [current_prediction] + forecast_prices}
        prediction_text = f"₹ {current_prediction:,.2f} Lakhs"
        deal_info = {'seller_price_text': f"₹ {seller_offer:,.2f} Lakhs"}
        if seller_offer > 0:
            if seller_offer > current_prediction * 1.1: deal_info.update({'status': 'Priced Above Market', 'message': 'Your asking price is higher than the AI-predicted market value.', 'color': 'warning'})
            elif seller_offer < current_prediction * 0.9: deal_info.update({'status': 'Priced Below Market', 'message': 'Your asking price is significantly below market value.', 'color': 'danger'})
            else: deal_info.update({'status': 'Fair Market Price', 'message': 'Your asking price is competitive.', 'color': 'success'})
        
        return render_template(
            template, 
            prediction_text=prediction_text, 
            deal=deal_info, 
            forecast=forecast_data, 
            vehicle_details=form_data,
            **dropdowns
        )
    except Exception as e:
        print(f"ERROR in /predict: {e}")
        return f"An error occurred: {e}", 400

# The __main__ block is not needed for Vercel, but it's fine to leave it for local testing.
if __name__ == '__main__':
    app.run(debug=True)