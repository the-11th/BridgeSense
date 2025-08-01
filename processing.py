import pandas as pd
import joblib

try:
    model_columns = joblib.load("model_columns.joblib")
except FileNotFoundError:
    print("There is no related file.")
    model_columns = []

def transform_data(input):
  
    if not model_columns:
        raise ValueError("No joblib file!")

    df = pd.DataFrame([input])

    age = df.get('Bridge Age (yr)', 0)
    traffic = df.get('Computed - Average Daily Truck Traffic (Volume)', 0)
    op_rating = df.get('64 - Operating Rating (US tons)', 0)

    df['Age_x_Traffic'] = age * traffic
    df['squared'] = age ** 2
    df['Age_x_OpRating'] = age * op_rating
    
    df_encoded = pd.get_dummies(df)

    df1 = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    return df1