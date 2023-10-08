import pandas as pd

def load_data(url):
    data = pd.read_csv(url)
    # Perform data preprocessing tasks (handling missing values, feature engineering, etc.)
    return data
