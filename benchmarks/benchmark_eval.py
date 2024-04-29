import pandas as pd
import json

# Read the CSV file
df = pd.read_csv('vivechan_evaluation_benchmark.csv')

# Create an empty list to store the JSON objects
json_list = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    question = row['Question']
    # Create a JSON object with the desired columns
    json_obj = {
        'No.': index + 1,
        'Question': row['Question'],
        'Scripture': row['Scripture']
    }
    # Append the JSON object to the list
    json_list.append(json_obj)

# Print the list of JSON objects
print(json.dumps(json_list))