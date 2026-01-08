from sklearn.preprocessing import StandardScaler
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from sklearn.metrics import mean_absolute_error


def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)


n1features = []
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)


app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/train')
def train():
    data = pd.read_csv('static/Train.csv')
    data = data.sort_values(
        by=['date_time'], ascending=True).reset_index(drop=True)
    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data = data.dropna().reset_index(drop=True)
    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)

    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
    data.to_csv("traffic_volume_data.csv", index=None)

    sns.set()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore')
    data = pd.read_csv("traffic_volume_data.csv")
    data = data.sample(10000, replace=True).reset_index(drop=True)
    label_columns = ['weather_type']
    numeric_columns = ['is_holiday', 'temperature',
                       'weekday', 'hour', 'month_day', 'year', 'month']
    n1 = data['weather_type']
    unique(n1)
    n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
                  'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    """#Data Preparation"""
    n11 = []
    for i in range(10000):
        if(n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i]))+1)
    data['weather_type'] = n11
    features = numeric_columns+label_columns
    target = ['traffic_volume']
    X = data[features]
    y = data[target]
    print(X)
    print(data[features].hist(bins=20,))

    data['traffic_volume'].hist(bins=20)

    """#Feature Scaling"""

    X = x_scaler.fit_transform(X)

    y = y_scaler.fit_transform(y).flatten()
    print(X)
    warnings.filterwarnings('ignore')

    regr.fit(X, y)
    # error eval
    from sklearn.model_selection import train_test_split
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))
    print('predicted output :=', regr.predict(X[:10]))
    print('Actual output :=', y[:10])
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ip = []

    # Handle holiday
    ip.append(1 if request.form['isholiday'] == 'yes' else 0)

    # Temperature conversion: Kelvin to Celsius
    temperature_kelvin = float(request.form['temperature'])
    ip.append(int(temperature_kelvin - 273.15))

    # Date and time handling
    ip.append(int(request.form['day']))
    ip.append(int(request.form['time'][:2]))

    D = request.form['date']
    ip.append(int(D[8:]))  # Day
    ip.append(int(D[:4]))  # Year
    ip.append(int(D[5:7]))  # Month

    # Weather features
    s1 = request.form.get('x0')

    ip.append((n1features.index(s1) + 1) if s1 in n1features else 0)

    # Prediction
    ip_scaled = x_scaler.transform([ip])
    out = regr.predict(ip_scaled)
    y_pred = y_scaler.inverse_transform([out])[0]

    # Interpret result
    if y_pred <= 1000:
        s = "No Traffic – You can go the same way."
    elif y_pred <= 3000:
        s = "Busy or Normal Traffic – You can still go the same way."
    elif y_pred <= 5500:
        s = "Heavy Traffic – Consider changing your route."
    else:
        s = "Worst Case – You must change your route."

    return render_template(
    'output.html',
    statement=s,
    op=int(y_pred),
    date=request.form['date'],
    day=request.form['day'],
    time=request.form['time'],
    temperature=request.form['temperature'],
    isholiday=request.form['isholiday'],
    x0=request.form['x0']
)




if __name__ == '__main__':
    app.run(debug=True)
