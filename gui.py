import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from weather_predictor import txtCheck, ltxtCheck, SVM_Model, ARIMA_MODEL, GradientBoostingModel, KNN_Model, display_bar_graph
from PIL import Image, ImageTk  # Import Pillow for image support
import matplotlib.pyplot as plt
# Read weather data from file
with open("weather.txt", "r") as file:
    contents = file.read()
    array = contents.split('.')
    lines = contents.splitlines()
    last_line = lines[-1].strip()
    lastarray = lines[-1].split(' ')

def check_weather():
    # Clear the previous graph
    for widget in graph_frame.winfo_children():
        widget.destroy()

    # Get weather data
    weather_data = txtCheck(array)
    if weather_data is not None:
        selected_model = model_dropdown.get()

        if selected_model == 'SVM Model':
            forecast = SVM_Model(weather_data)
            temperature = int(forecast[0])
            humidity = int(forecast[1])
            wind = int(forecast[2])
            title = 'SVM Model'
            fig = display_bar_graph(title, temperature, wind, humidity, ['red', 'blue', 'green'])

        elif selected_model == 'ARIMA Model':
            forecast = ARIMA_MODEL(weather_data)
            temperature = int(forecast[0])
            humidity = int(forecast[1])
            wind = int(forecast[2])
            title = 'ARIMA Model'
            fig = display_bar_graph(title, temperature, wind, humidity, ['red', 'blue', 'green'])

        elif selected_model == 'Gradient Boosting Model':
            forecast = GradientBoostingModel(weather_data)
            temperature = int(forecast[0])
            humidity = int(forecast[1])
            wind = int(forecast[2])
            title = 'Gradient Boosting Model'
            fig = display_bar_graph(title, temperature, wind, humidity, ['red', 'blue', 'green'])

        elif selected_model == 'K-NN Model':
            x_train = np.array(weather_data)[:, :2]  # Features: Temperature & Humidity
            y_train = np.array(weather_data)[:,:]  # Only Temp, Wind, Humidity

            
            res = ltxtCheck(lastarray)  # Ensure this returns [wind_speed, humidity]
            x_test = np.array([res])  # Convert input to NumPy array
            
            forecast = KNN_Model(x_train, y_train, x_test, 3)  # Predict all three values
            temperature, humidity,wind_speed = forecast[0]  # Extract predictions

            title = 'K-NN Model'
            fig = display_bar_graph(title, int(temperature), int(wind_speed), int(humidity), ['red', 'blue', 'green'])


        elif selected_model == 'All':
            # Get forecasts from all models
            forecast_svm = SVM_Model(weather_data)
            forecast_arima = ARIMA_MODEL(weather_data)
            forecast_gbm = GradientBoostingModel(weather_data)
            x_train = np.array(weather_data)[:, :2]  # Features: Temperature & Humidity
            y_train = np.array(weather_data)[:,:]  # Only Temp, Wind, Humidity
            res = ltxtCheck(lastarray)
            x_test = np.array([res])
            forecast_knn = KNN_Model(x_train, y_train, x_test, 3)

            # Extract values and convert to integers
            temp_svm, humid_svm, wind_svm = map(int, forecast_svm)  # Corrected: Use `int` instead of `'int'`
            temp_arima, humid_arima, wind_arima = map(int, forecast_arima)
            temp_gbm, humid_gbm, wind_gbm = map(int, forecast_gbm)
            temp_knn, humid_knn, wind_knn = map(int, forecast_knn[0])

            # Create a grouped bar plot
            fig, ax = plt.subplots()
            labels = ['Temperature', 'Wind', 'Humidity']
            x = np.arange(len(labels))
            width = 0.2

            rects1 = ax.bar(x - 1.5 * width, [temp_svm, wind_svm, humid_svm], width, label='SVM Model', color='red')
            rects2 = ax.bar(x - 0.5 * width, [temp_arima, wind_arima, humid_arima], width, label='ARIMA Model', color='blue')
            rects3 = ax.bar(x + 0.5 * width, [temp_gbm, wind_gbm, humid_gbm], width, label='Gradient Boosting Model', color='green')
            rects4 = ax.bar(x + 1.5 * width, [temp_knn, wind_knn, humid_knn], width, label='K-NN Model', color='yellow')

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Values')
            ax.set_title('Comparison of ALL Models')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            ax.bar_label(rects1, padding=3)
            ax.bar_label(rects2, padding=3)
            ax.bar_label(rects3, padding=3)
            ax.bar_label(rects4, padding=3)

            fig.tight_layout()

        else:
            return

        # Display the new graph
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

# GUI Setup
window = tk.Tk()
window.title("Weather Predictor")
window.geometry("600x500")

# Load the background image using Pillow and resize it to fit the window
try:
    pil_image = Image.open("bg.png")
    # Resize the image to fit the window size
    pil_image = pil_image.resize((600, 500), Image.ANTIALIAS)
    background_image = ImageTk.PhotoImage(pil_image)
    background_label = tk.Label(window, image=background_image)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Error loading background image: {e}")

# Heading
font_heading = ("Helvetica", 16, "bold")
heading = tk.Label(window, text="WEATHER PREDICTOR", font=font_heading, bg='white')
heading.pack(pady=10)

# Model Dropdown
model_label = tk.Label(window, text="Select Prediction Model:", font=("Helvetica", 12), bg='white')
model_label.pack(pady=10)
model_options = ['SVM Model', 'ARIMA Model', 'Gradient Boosting Model', 'K-NN Model', 'All']
model_dropdown = ttk.Combobox(window, values=model_options)
model_dropdown.pack()

# Weather Check Button
check_button = tk.Button(window, text="Predict", command=check_weather, font=font_heading)
check_button.pack(pady=7)

# Graph Frame
graph_frame = tk.Frame(window, bg='white')
graph_frame.pack(fill=tk.BOTH, expand=True)

window.mainloop()