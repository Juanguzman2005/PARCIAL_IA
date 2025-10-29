
import pandas as pd

def load_csv_from_url(url):
    df = pd.read_csv(url)
    return df
