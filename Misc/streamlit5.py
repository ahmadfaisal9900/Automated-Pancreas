import torch
import torch.nn as nn
import streamlit as st

st.title("Automated Pancreas")
class PrescriptiveModel(nn.Module):
    def __init__(self, input_size):
        super(PrescriptiveModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 50)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(50, 90)
        self.relu = nn.ReLU()
        self.linear4 = nn.Linear(90, 30)
        self.relu = nn.ReLU()
        self.linear5 = nn.Linear(30, 10)
        self.relu = nn.ReLU()
        self.linear6 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.relu(self.linear5(x))
        x = self.relu(self.linear6(x))
        return x

prescriptive_model = PrescriptiveModel(4)

# Load the trained weights into the model architecture
prescriptive_model = torch.load("prescriptive_model.pth")
predictive_model = nn.Sequential(
    nn.Linear(3, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# Load the checkpoint
checkpoint = torch.load("model.pth")

# Load the model parameters from the checkpoint
predictive_model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
predictive_model.eval()

def predict_insulin_dosage(prescriptive_model, input_data):
    """
    Predicts insulin dosage based on glucose level and meal size.
    """
    inputs = torch.tensor([input_data]).float()
    print(inputs)
    predicted_dose = prescriptive_model(inputs)
    predicted_dose = predicted_dose/1000
    return predicted_dose.item()  # Extract the actual value from the tensor

def predict_glucose_level(descriptive_model, glucose_level, insulin_dose, meal_size):
    """
    Predicts glucose level based on insulin dosage and meal size.
    """
    inputs = torch.tensor([glucose_level, meal_size, insulin_dose]).float()
    predicted_glucose = descriptive_model(inputs)
    return predicted_glucose.item()  # Extract the actual value from the tensor
import random
import warnings
import numpy as np
from sklearn.decomposition import PCA
from joblib import load


warnings.filterwarnings("ignore")
pca = load('pca_model.joblib')
initial_value = (150, 0)
glucose_level = st.number_input('Initial Glucose Level', min_value=0, max_value=500, value=150)
meal_size = st.number_input('Initial Meal Size', min_value=0, max_value=100, value=0)
num_iterations = st.slider('Number of Iterations', min_value=100, max_value=1000, step=100)




glucose_levels = []
insulin_dosages = []

for i in range(num_iterations):
    # Administer a meal every 100 iterations
    if i % 100 == 0:
        meal_size = random.randint(50,100)
    else:
        meal_size = 0

    # Transform the glucose level and meal size using the trained PCA model
    new_data = np.array([[glucose_level, meal_size]])
    new_data_transformed = pca.transform(new_data)
    new_data_pca = np.append(new_data, new_data_transformed)

    # Use prescriptive model to predict insulin dosage
    if glucose_level > 85:
        predicted_dose = predict_insulin_dosage(prescriptive_model, new_data_pca)
        print(f"Insulin Dosage {predicted_dose}")
    else:
        predicted_dose = 0

    # Store glucose level and insulin dosage
    glucose_levels.append(glucose_level)
    insulin_dosages.append(predicted_dose)
    # Use descriptive model to predict resulting glucose level
    # (assuming the predicted dose is used)
    glucose_level = predict_glucose_level(predictive_model, glucose_level, predicted_dose, meal_size)

import matplotlib.pyplot as plt
# Plot the results
# Calculate average insulin dosage
average_insulin_dosage = np.mean(insulin_dosages)

# Plot glucose levels
st.subheader('Glucose Levels Over Time')
fig_glucose, ax_glucose = plt.subplots(figsize=(12, 6))
ax_glucose.plot(glucose_levels, label='Glucose Level', color='blue')
ax_glucose.set_xlabel('Time (iterations)')
ax_glucose.set_ylabel('Glucose Level (mg/dL)')
ax_glucose.legend()
st.pyplot(fig_glucose)

# Plot insulin dosages
st.subheader('Insulin Dosages Over Time')
fig_insulin, ax_insulin = plt.subplots(figsize=(12, 6))
ax_insulin.plot(insulin_dosages, label='Insulin Dosage', color='orange')
ax_insulin.set_xlabel('Time (iterations)')
ax_insulin.set_ylabel('Insulin Dosage (units)')
ax_insulin.legend()
st.pyplot(fig_insulin)

# Display predictions
st.subheader('Average Insulin Dosage')
st.write(f"The average insulin dosage is: {average_insulin_dosage:.3f} units")

st.subheader('Final Glucose Level')
# Assuming `glucose_level` is defined earlier in your code
final_glucose_level = glucose_levels[-1]  # Example: replace with your final glucose level
st.write(f"The final predicted glucose level is: {final_glucose_level:.3f} mg/dL")