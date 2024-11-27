import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import os
def preprocess(file_path, scen, iter, output_dir):

  df = pd.read_csv(file_path)
  columns_to_keep = ['Time', 'CGM', 'insulin']

  all_columns = df.columns.tolist()

  columns_to_drop = [col for col in all_columns if col not in columns_to_keep]

  df = df.drop(columns=columns_to_drop)

  df['Time'] = df['Time'].astype(str)

  df[['Date', 'Time']] = df['Time'].str.split(' ', expand=True)

  meal_df = pd.DataFrame(scen, columns=['Hour', 'Meal'])
  

  meal_df['Hour'] = meal_df['Hour'].apply(lambda x: f'{x:02}:03:00')

  # Add a new column named 'Meal' to the DataFrame
  df['Meal'] = 0

  # Assuming 'Hour' is a string column with consistent time format

  hour_list = meal_df['Hour'].tolist()
  meal_list = meal_df['Meal'].tolist()

  # Iterate through 'Time' column
  for index, time in df['Time'].items():
    # No need to convert 'time' (assuming it's already a datetime object)
    time_str = time  # time_str already contains the formatted time string

    # Check if time string exists in hour_list
    if time_str in hour_list:
      # Get the corresponding meal index (assuming order is consistent)
      meal_index = hour_list.index(time_str)
      
      # Update 'Meal' column value at the current index
      df.at[index, 'Meal'] = meal_list[meal_index]

  # Convert the 'Date' column to datetime format
  df['Date'] = pd.to_datetime(df['Date']).dt.date

  # Convert the 'Time' column to time format
  df['Time'] = pd.to_datetime(df['Time']).dt.time
  # Create a new column 'Prediction' and shift the 'CGM' column up by one row
  df['Prediction'] = df['CGM'].shift(-1)

  # Print the last few rows of the DataFrame
  

  output_file_name = f'adolescents{iter+20}_processed.csv'
  df.to_csv(os.path.join(output_dir, output_file_name), index=False)

