import pandas as pd
import joblib

# --- 加载数据转换所需的“资产” ---
# 这个文件依赖于 'model_columns.joblib'，该文件在训练时生成。
try:
    model_columns = joblib.load("model_columns.joblib")
except FileNotFoundError:
    # 这是一个后备方案，但在实际运行时，文件必须存在
    print("There is no file of 'model_columns.joblib'.")
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