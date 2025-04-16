import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def preprocess_input(area, bedrooms, bathrooms, basement, hotwaterheating, airconditioning, parking, furnishingstatus):
    user_input = {'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                  'basement': 1 if basement.lower() == 'yes' else 0,
                  'hotwaterheating': 1 if hotwaterheating.lower() == 'yes' else 0,
                  'airconditioning': 1 if airconditioning.lower() == 'yes' else 0,
                  'parking': parking,
                  'furnishingstatus': {'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0}.get(furnishingstatus.lower(), 0)}
    input_data = pd.DataFrame(user_input, index=[0])
    return input_data

def predict_price(model, input_data):
    predicted_price = model.predict(input_data)
    return predicted_price[0]

def on_predict_button_click(*args):
    user_input_data = preprocess_input(
        float(entries[0].get()), 
        int(entries[1].get()), 
        int(entries[2].get()), 
        entries[3].get(), 
        entries[4].get(), 
        entries[5].get(), 
        int(entries[6].get()), 
        entries[7].get()
    )
    predicted_price = predict_price(model, user_input_data)
    result_label.config(text=f'Predicted Price: ${predicted_price:.2f}')

    plot_model()

def plot_model():
    area_values, y_pred_values = X_test['area'], model.predict(X_test)

    plt.scatter(area_values, y_test, label='Actual Values')
    plt.plot(area_values, y_pred_values, linewidth=2, color='red', label='Linear Regression Line')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Values with Linear Regression Line')
    plt.legend()

    for widget in frame.winfo_children():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=10, padx=10)

data = pd.read_csv('Housing.csv').drop(['prefarea', 'stories'], axis=1)
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'furnishingstatus']
data[categorical_columns] = data[categorical_columns].replace({'no': 0, 'yes': 1, 'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0})

selected_features = ['area', 'bedrooms', 'bathrooms', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'furnishingstatus']
X, y = data[selected_features], data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)

root = tk.Tk()
root.title("House Price Prediction")

root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

style = ttk.Style().theme_use("clam")

frame = ttk.Frame(root, padding=(10, 10, 10, 10))
frame.grid(column=0, row=0)

labels = ['Area:', 'Bedrooms:', 'Bathrooms:', 'Basement (yes/no):', 'Hot Water Heating (yes/no):',
          'Air Conditioning (yes/no):', 'Parking:', 'Furnishing Status (furnished/semi-furnished/unfurnished):']

entry_vars = [tk.StringVar() for _ in range(len(labels))]
entries = [ttk.Entry(frame, textvariable=var) for var in entry_vars]

for i, label in enumerate(labels):
    ttk.Label(frame, text=label, foreground="green").grid(row=i, column=0, padx=5, pady=5)
    entries[i].grid(row=i, column=1, padx=5, pady=5, sticky="ew")

predict_button = ttk.Button(frame, text="Predict", command=on_predict_button_click)
predict_button.grid(row=8, column=0, columnspan=2, pady=10)

result_label = ttk.Label(frame, text="")
result_label.grid(row=9, column=0, columnspan=2, pady=5)

frame.rowconfigure(0, weight=1)
frame.columnconfigure(0, weight=1)
frame.columnconfigure(2, weight=1)

root.bind("<Return>", on_predict_button_click)

root.mainloop()