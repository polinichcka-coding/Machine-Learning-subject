import pickle
import numpy as np
import pandas as pd


FEATS = [
    'Location', 'Bedrooms', 'Bathrooms',
    'Floor Area', 'Land Area', 'Latitude', 'Longitude'
]

# Load the model and preprocessing information from the pickle file
def apply_prep(df, prep):
    medians, modes, bounds, zones = prep
    df = df.copy()

    for col, value in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    for col, value in modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)

    for col, (low, high) in bounds.items():
        if col in df.columns and col != 'Price':
            df[col] = df[col].clip(low, high)

    # Apply the same geographic zones used during training.
    lat_edges, lon_edges = zones
    lat = pd.to_numeric(df['Latitude'], errors='coerce').to_numpy()
    lon = pd.to_numeric(df['Longitude'], errors='coerce').to_numpy()
    lat_group = np.searchsorted(lat_edges, lat, side='right')
    lon_group = np.searchsorted(lon_edges, lon, side='right')
    df['Location'] = [f'Zone_{a+1}_{b+1}' for a, b in zip(lat_group, lon_group)]
    return df


# Walk through the decision tree to make a prediction for a single row of data
def walk(node, row):
    while node['feat'] is not None:
        value = row[node['feat']]

        if node['thr'] is not None:
            key = 'le' if value <= node['thr'] else 'gt'
        else:
            key = value if value in node['kids'] else '__other__'

        if key not in node['kids']:
            break

        node = node['kids'][key]

    return node['value']


def predict_price(obj):
    with open('final_model.pkl', 'rb') as file:
        model = pickle.load(file)

    df = pd.DataFrame([obj])
    row = apply_prep(df, model['prep']).iloc[0].to_dict()

    predictions = [
        walk(tree, row)
        for tree in model['trees']
    ]

    return float(np.mean(predictions))


if __name__ == '__main__':
    print("Enter house information:")

    obj = {}

    for feature in FEATS:
        value = input(f'{feature}: ').strip()

        if feature == 'Location':
            obj[feature] = value if value else None
        else:
            obj[feature] = float(value) if value else np.nan

    price = predict_price(obj)

    print(f'Predicted price: {price:,.0f} PHP')
