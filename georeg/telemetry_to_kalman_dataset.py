import os
import json

def process_telemetry_files(input_dir, output_file):
    measurements = []

    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        if filename.startswith('telemetry_') and filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            # Load the telemetry data from the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract required fields and format them
            measurement = {
                "lat": data["lat"],
                "lon": data["lon"],
                # Add other fields if needed
            }
            measurements.append(measurement)

    # Save the combined measurements to the output file
    with open(output_file, 'w') as f:
        json.dump(measurements, f, indent=4)

# Define input directory and output file
input_dir = '/home/yossi/Documents/database/videos/drones/drone_pov/geotsvideo/images_with_klv/telemetry/'
output_file = 'telemetry_measurements.json'

# Process the telemetry files and create the combined JSON
process_telemetry_files(input_dir, output_file)
