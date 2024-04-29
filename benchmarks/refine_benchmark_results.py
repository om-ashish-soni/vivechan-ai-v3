import json

# Read benchmark_result.json
with open('benchmark_result.json', 'r') as file:
    data = json.load(file)

# Format the JSON data
formatted_data = json.dumps(data, indent=4)

# Print the formatted data
print(formatted_data)

# Dump the formatted data into benchmark_result_formatted.json
with open('benchmark_result_formatted.json', 'w') as file:
    file.write(formatted_data)