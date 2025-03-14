import numpy as np
from textblob import TextBlob
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt

def txtCheck(array):
    weather = []
    for i in array:
        temp_data = None
        humidity_data = None
        wind_data = None
        blob = TextBlob(i)
        for word, tag in blob.tags:
            if tag == 'CD' and '°C' in word:
                temp_data = int(word.replace('°C', '').replace('Â', ''))
            if 'Humidity:' in word:
                try:
                    humidity_data = int(word.replace('Humidity:', ''))
                except ValueError:
                    print(f"Invalid humidity value: {word}")
            if tag == 'CD' and 'km/h' in word:
                wind_data = int(word.replace('km/h', ''))
        if temp_data is not None and humidity_data is not None and wind_data is not None:
            weather.append([temp_data, humidity_data, wind_data])
    return weather

def ltxtCheck(array):
    temperature = []
    humidity = []
    winds = []
    for i in array:
        blob = TextBlob(i)
        for word, tag in blob.tags:
            if tag == 'CD' and '°C' in word:
                temperature.append(int(word.replace('°C', '').replace('Â', '')))
            if 'Humidity:' in word:
                try:
                    humidity.append(int(word.replace('Humidity:', '')))
                except ValueError:
                    print(f"Invalid humidity value: {word}")
            if tag == 'CD' and 'km/h' in word:
                winds.append(int(word.replace('km/h', '')))
    return temperature, humidity, winds

class SimpleGradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.01):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        self.models = []
        y_pred = np.zeros_like(y, dtype=float)
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            model = [np.polyfit(X.flatten(), residuals[:, i], 1) for i in range(residuals.shape[1])]
            y_pred += self.learning_rate * np.array([np.polyval(m, X.flatten()) for m in model]).T
            self.models.append(model)

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.models[0])), dtype=float)
        for model in self.models:
            y_pred += self.learning_rate * np.array([np.polyval(m, X.flatten()) for m in model]).T
        return y_pred

class SimpleARModel:
    def __init__(self, p):
        self.p = p
        self.coefs_ = []

    def fit(self, y):
        self.coefs_ = []
        for i in range(y.shape[1]):
            yi = y[:, i]
            X = np.column_stack([np.roll(yi, j) for j in range(1, self.p + 1)])
            X = X[self.p:]
            yi = yi[self.p:]
            coef = np.linalg.lstsq(X, yi, rcond=None)[0]
            self.coefs_.append(coef)

    def predict(self, y, n_steps):
        predictions = np.zeros((n_steps, y.shape[1]))
        for step in range(n_steps):
            X = y[-self.p:]
            for i, coef in enumerate(self.coefs_):
                pred = np.dot(X[:, i], coef)
                predictions[step, i] = pred
                y = np.append(y, [[pred] * y.shape[1]], axis=0)
        return predictions

class LinearSVR:
    def __init__(self, C=1e3):
        self.C = C
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for i in range(y.shape[1]):
            yi = y[:, i]
            K = np.dot(X, X.T)
            P = matrix(np.vstack([np.hstack([K, -K]), np.hstack([-K, K])]), tc='d')
            q = matrix(self.C * np.ones(2 * n_samples) - np.hstack([yi, -yi]), tc='d')
            G = matrix(np.vstack([-np.eye(2 * n_samples), np.eye(2 * n_samples)]), tc='d')
            h = matrix(np.hstack([np.zeros(2 * n_samples), np.ones(2 * n_samples) * self.C]), tc='d')
            A = matrix(np.hstack([np.ones(n_samples), -np.ones(n_samples)]).reshape(1, -1), tc='d')
            b = matrix(0.0, tc='d')
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)
            alphas = np.array(solution['x']).flatten()
            w = np.dot((alphas[:n_samples] - alphas[n_samples:]), X)
            support_indices = np.where((alphas[:n_samples] - alphas[n_samples:]) != 0)[0]
            b = np.mean(yi[support_indices] - np.dot(X[support_indices], w))
            self.models.append((w, b))

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], len(self.models)), dtype=float)
        for i, (w, b) in enumerate(self.models):
            y_pred[:, i] = np.dot(X, w) + b
        return y_pred

def SVM_Model(weather):
    if weather is not None and len(weather) > 0:
        X = np.array(range(len(weather))).reshape(-1, 1)
        y = np.array(weather)
        model = LinearSVR(C=1e3)
        model.fit(X, y)
        forecast = model.predict(np.array([[len(weather)]]))[0]
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

def ARIMA_MODEL(weather):
    if weather is not None and len(weather) > 0:
        y = np.array(weather)
        model = SimpleARModel(p=5)
        model.fit(y)
        forecast = model.predict(y, 1)[0]
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

def GradientBoostingModel(weather):
    if weather is not None and len(weather) > 0:
        X = np.array(range(len(weather))).reshape(-1, 1)
        y = np.array(weather)
        model = SimpleGradientBoostingRegressor()
        model.fit(X, y)
        forecast = model.predict(np.array([[len(weather)]]))[0]
        return forecast
    else:
        print("THE DATA IS NOT AVAILABLE FOR FORECASTING")
        return None

def _euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Euclidean Distance Function
def _euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN Model for Predicting Temperature, Wind Speed, and Humidity
def KNN_Model(x_train, y_train, X_test, k):
    y_pred = []  # Store predictions
    
    X_test = np.array(X_test)  # Ensure X_test is an array
    
    for test_point in X_test:
        distances = np.array([_euclidean_distance(test_point, train_point) for train_point in x_train])
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:k]  
        k_nearest_values = y_train[k_indices]  # Get corresponding Temperature, Wind, Humidity
        
        # Compute prediction (average for regression)
        prediction = np.mean(k_nearest_values, axis=0)  
        y_pred.append(prediction)
    
    return np.array(y_pred)  # Return predictions as a NumPy array

def display_bar_graph(title, temperature, wind, humidity, colors):
    labels = ['Temperature', 'Wind', 'Humidity']
    values = [temperature, wind, humidity]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, values, color=colors, width=0.4)
    ax.set_title(title)
    ax.bar_label(bars)
    return fig

def model_satisfaction_check(allTemp, allHumidty, allWind, temperature, humidity, Winds):
    model1count = 0
    model2count = 0
    model3count = 0
    model4count = 0

    print("\n================== MODEL SATISFACTION CHECK ==================")

    while True:
        if allTemp[0] == temperature[0]:
            print("Temperature Satisfied By SVM MODEL")
            model1count += 1
            if allWind[0] == Winds[0]:
                print("Wind Satisfied By SVM MODEL")
                model1count += 1
                if allHumidty[0] == humidity[0]:
                    print("Humidity Satisfied By SVM MODEL")
                    model1count += 1
        
        elif allTemp[1] == temperature[0]:
            print("Temperature Satisfied By ARIMA MODEL")
            model2count += 1
            if allWind[1] == Winds[0]:
                print("Wind Satisfied By ARIMA MODEL")
                model2count += 1
                if allHumidty[1] == humidity[0]:
                    print("Humidity Satisfied By ARIMA MODEL")
                    model2count += 1

        elif allTemp[2] == temperature[0]:
            print("Temperature Satisfied By GRADIENT BOOSTING MODEL")
            model3count += 1
            if allWind[2] == Winds[0]:
                print("Wind Satisfied By GRADIENT BOOSTING MODEL")
                model3count += 1
                if allHumidty[2] == humidity[0]:
                    print("Humidity Satisfied By GRADIENT BOOSTING MODEL")
                    model3count += 1

        elif allTemp[3] == temperature[0]:
            print("Temperature Satisfied By k-NN Model")
            model4count += 1
            if allWind[3] == Winds[0]:
                print("Wind Satisfied By k-NN Model")
                model4count += 1
                if allHumidty[3] == humidity[0]:
                    print("Humidity Satisfied By k-NN Model")
                    model4count += 1

        print(model1count, model2count, model3count, model4count)

        if model1count == 3:
            print("\nSVM MODEL PREDICTING WEATHER CORRECTLY")
            break   

        if model2count == 3:
            print("\nARIMA MODEL PREDICTING WEATHER CORRECTLY")
            break   

        if model3count == 3:
            print("\nGRADIENT MODEL PREDICTING WEATHER CORRECTLY")
            break

        if model4count == 3:
            print("\nKN-N MODEL PREDICTING WEATHER CORRECTLY")
            break 
        
        else:
            print("\nModels gave us the approximate Prediction but not accurate. ")

            if model1count > model2count or model1count > model3count or model1count > model4count:
                print("SVM MODEL PREDICTING THE RESULT BETTER AS COMPARE TO GRADIENT, ARIMA AND KN-N")
            if model2count > model1count or model2count > model3count or model2count > model4count:
                print("ARIMA MODEL PREDICTING THE RESULT BETTER AS COMPARE TO GRADIENT, SVM AND KN-N")
            if model3count > model2count or model3count > model1count or model3count > model4count:
                print("GRADIENT BOOSTING MODEL PREDICTING THE RESULT BETTER AS COMPARE TO SVM, ARIMA AND KN-N")
            if model4count > model1count or model4count > model2count or model4count > model3count:
                print("KN-N MODEL PREDICTING THE RESULT BETTER AS COMPARE TO SVM, ARIMA AND GRADIENT")
            break

def measure(temp, windss, humid):
    if temp > 25:
        if windss >= 15:
            if humid > 40 and humid < 70:
                print("The weather would be Hot and Sunny.")
                print("The chances of rain are low.")
                return
            else:
                print("The weather would be Hot and Sunny.")
                print("The chances of rain are a little bit HIGH.")
                return
            
    elif (temp > 15 and temp < 25) and (windss >= 5 and windss < 20) and (humid >= 70):
        print("It is a mild temperature, moderate wind, and relatively humid.")
        print("The chances of rain are moderate.")
        return

    elif (temp <= 15) and (windss >= 20) and (humid >= 70):
        print("The weather would be cold, windy, and have high humidity.")
        print("The chances of rain are high.")
        return
    
    elif (temp <= 15) and (windss >= 20):
        print("The weather would be cold and windy.")
        print("The chances of rain are low to moderate.")
        return
    
    elif (temp <= 15) and (humid >= 70):
        print("The weather would be cold and humid.")
        print("The chances of rain are moderate to high.")
        return
    
    elif (windss >= 20) and (humid >= 70):
        print("The weather would be windy and humid.")
        print("The chances of rain are moderate to high.")
        return
    
    elif (temp > 25) and (windss <= 5) and (40 <= humid <= 70):
        print("The weather would be Hot with moderate humidity.")
        print("The chances of rain are low to moderate.")
        return
    
    else:
        print("No specific condition matched for weather or rain chances.")
        return