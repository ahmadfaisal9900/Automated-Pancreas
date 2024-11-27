import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import seaborn as sns
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess
import os
# Define the path to the directory with the CSV files
input_dir = 'seeds\\seed3'
output_dir = 'processed_results'

# Check if output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Define the scenario
data = [
    [(8, 55), (12, 80), (16, 40), (18, 50), (23, 75)]
,[(7, 80), (11, 75), (15, 20), (18, 35), (22, 25)]
,[(8, 45), (12, 30), (14, 90), (18, 50), (22, 45)]
,[(7, 55), (11, 80), (15, 45), (17, 60), (22, 35)]
,[(7, 60), (12, 90), (16, 65), (18, 20), (22, 30)]
,[(8, 65), (11, 75), (15, 35), (18, 45), (22, 25)]
,[(7, 20), (12, 80), (15, 30), (18, 75), (22, 60)]
,[(8, 50), (12, 70), (14, 35), (18, 80), (22, 30)]
,[(7, 40), (11, 65), (15, 50), (18, 95), (22, 45)]
,[(7, 25), (12, 80), (14, 35), (18, 45), (22, 95)]
]

# Loop over all CSV files
for iter, file_name in enumerate(csv_files, start=1):
    #print(data[iter-1])
    file_path = os.path.join(input_dir, file_name)
    scen = data[iter-1]
    # Call the preprocessing function
    preprocess(file_path, scen, iter, output_dir)


import pandas as pd
import os

# Define the path to the directory with the CSV files
input_dir = 'processed_results'

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

# Initialize an empty list to hold the DataFrames
dfs = []

# Loop over all CSV files
for file_name in csv_files:
    # Define the file path
    file_path = os.path.join(input_dir, file_name)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
df = pd.concat(dfs, ignore_index=True)


#df = pd.read_csv('E:\\Projects\\Unfinished\\Insulin Project\\results2\\adolescent#001_processed.csv')
df.drop(['Time', 'Date'], axis=1, inplace=True)
df.dropna(inplace=True)

insulin_threshold = 1

df.loc[df['insulin'] > insulin_threshold, 'Meal'] = 1

df['Insulin_Spike'] = (df['Meal'] == 1) & (df['insulin'] > insulin_threshold)

X = df[['CGM', 'Meal', 'insulin', 'Insulin_Spike']].values
y = df['Prediction'].values


scaler = StandardScaler()
X = scaler.fit_transform(X)  # Normalize the features

# Create sequences of data
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
X_seq, y_seq = create_sequences(X, y, seq_length)

# Count the number of non-zero values in the 'insulin' column
non_zero_insulin_count = (df['insulin'] != 0).sum()
total_insulin_count = (df['insulin'] == 0).sum()
# Print the result
print(f'Number of non-zero insulin values: {non_zero_insulin_count}')
print(f'Number of zero insulin values: {total_insulin_count}')
# Split the data into training and testing sets
# # Split the data into training and testing sets, while also getting the indices
x = 14400 - seq_length
df_index = df.index[0:x]
print(df_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"X_seq: {X_seq.shape}")
print(f"Y_Seq {y_seq.shape}")
X_train, X_test, y_train, y_test, train_indices, test_indices= train_test_split(
    X_seq, y_seq, df_index, test_size=0.2, random_state=42, shuffle=False
)

# Convert the data into PyTorch tensors
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)
print("Training features mean:", X_train.mean())
print("Training features std:", X_train.std())

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        
        # Define the LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define fully connected layers as a sequential module
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h_0, c_0))
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Pass through the fully connected layers
        out = self.fc_layers(out)
        
        return out


# Model parameters
input_size = 4
hidden_size = 50
output_size = 1
num_layers = 2

# Instantiate the model
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

model.apply(init_weights)

# Define the loss function and the optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# Define the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # Decays by 1% every epoch
# Train the model
epochs = 1000
from tqdm import tqdm
for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred.squeeze(), y_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    if epoch % 10 == 0:  # Print every 10 epochs
        print(f'Epoch {epoch}, Loss: {loss.item()}')

import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)

# Convert tensors to numpy arrays for plotting
y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()  # Move to CPU, detach, and flatten
y_test_np = y_test.cpu().detach().numpy()  # Move to CPU and detach


print(f"Predictions shape {y_pred_test_np.shape}")
print(f"Actual shape {y_pred_test.shape}")

print("First few actual values:", y_test_np[:10])
print("First few predicted values:", y_pred_test_np[:10])

# Print the values right before plotting
print("Values to be plotted:")
print("Actual values:", y_test_np)
print("Predicted values:", y_pred_test_np)

# Create a scatter plot of actual vs predicted values
plt.scatter(y_test_np, y_pred_test_np)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
# Compute the loss on the test set
test_loss = criterion(y_pred_test.squeeze(), y_test)
print(f'Test Loss: {test_loss.item()}')

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'Predictive_Model_LSTM.pth')

# Step 1: Define the threshold for heavy insulin dose
insulin_threshold = 1  # Change this as per your requirement

# Step 2: Calculate the glucose trend
# Create a new column 'Glucose_Trend' based on the difference between the current and previous glucose values
df['Glucose_Trend'] = df['CGM'].diff().fillna(0)  # Use the difference method to find the change

# Step 3: Add a column indicating whether the insulin dose was heavy
df['Heavy_Insulin'] = df['insulin'] > insulin_threshold

# Create a new DataFrame with actual and predicted values along with insulin levels
test_df = pd.DataFrame({
    'Actual': y_test_np,
    'Predicted': y_pred_test_np,
    'Insulin': df.loc[test_indices, 'insulin'].values,
    'Glucose_Trend': df.loc[test_indices, 'Glucose_Trend'].values,
    'Heavy_Insulin': df.loc[test_indices, 'Heavy_Insulin'].values
})


# Define the insulin threshold
insulin_threshold = 1  # Adjust based on what you consider "high" insulin

# Make predictions on the test set
model.eval()
with torch.no_grad():
    y_pred_test = model(X_test)



def sensitivity_analysis(model, X, feature_idx, num_points=100):
    model.eval()
    
    # Get the range of the feature
    feature_min = X[:,:,feature_idx].min()
    feature_max = X[:,:,feature_idx].max()
    feature_min_np = feature_min.cpu().detach().numpy()
    feature_max_np = feature_max.cpu().detach().numpy()
    # Create a range of values to test
    test_range = np.linspace(feature_min_np, feature_max_np, num_points)
    
    results = []
    
    with torch.no_grad():
        for value in test_range:
            X_modified = X.clone()
            X_modified[:,:,feature_idx] = value
            output = model(X_modified)
            results.append(output.mean().item())
    
    return test_range, results

# Assuming X_test is your test data
feature_names = ['CGM', 'Meal', 'insulin', 'Insulin_Spike']


for i, feature in enumerate(feature_names):
    x_vals, y_vals = sensitivity_analysis(model, X_test, i)
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals)
    plt.title(f'Sensitivity Analysis for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Model Output')
    plt.show()
