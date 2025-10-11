
from birdnetlib.batch import DirectoryAnalyzer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pprint import pprint
import os
import math



#this will only retrieve detections that are in the species list (one species- the Savannah Sparrow)
custom_list_path = "species_list.txt"  # See example file for formatting.
#two species we are interested in, but for now just do Savannah Sparrow
#Passerculus sandwichensis_Savannah Sparrow
#Dolichonyx oryzivorus_Bobolink

def on_analyze_complete(recording):
    print(recording.path)
    pprint(recording.detections)


def on_error(recording, error):
    print("An exception occurred: {}".format(error))
    print(recording.path)

#Compute the average amplitude of a set of samples (won't work til we get amplitude working)
def eval_amplitude(samples):
  max=-1
  sum=0.0
  for s in samples:
    #print(s)
    amp=s*s
    if amp > max:
      max=amp
    sum=sum+amp
  return(max,math.sqrt(sum/len(samples)),math.sqrt(sum))

""
print("Starting Analyzer")
analyzer = Analyzer(custom_species_list_path=custom_list_path)


#James:
#The data will have a root directory containing a directory for each sensor
#Within those directories will a directory for each day containing the files for each day (each file is about an hour of recording)
#The results should be:
#For each sensor, for each day: a sum amplitude for that day You can store this in a 2D array or a hashmap (sensor: hashmap(day:sum_amplitude))

#for now, to find the sum amplitude, just assume the amplitude of each file is 1.0. 
#for each sample, multiply amplitude * the confidence of the sample. Sum all the samples for a day directory together to get the total sum
#later we will figure out how to get the actual amplitude (I have a way using my own analysis, but it doesn't use birdnet)

#you will need to call the following code multiple times of course for each day directory in each sensor directory.
# Step 2: Process all directories and calculate sum amplitudes
base_directory = "./ShelburneSubset"

# Dictionary to store results: {sensor: {date: sum_amplitude}}
results = {}

print("\n=== Processing All Sensor-Day Combinations ===\n")

# Get all sensor directories
sensor_dirs = [d for d in os.listdir(base_directory) 
               if os.path.isdir(os.path.join(base_directory, d)) and d.startswith("Sensor")]

# Loop through each sensor
for sensor in sorted(sensor_dirs):
    sensor_path = os.path.join(base_directory, sensor)
    
    # Initialize dictionary for this sensor
    results[sensor] = {}
    
    # Get all date directories in this sensor
    date_dirs = [d for d in os.listdir(sensor_path) 
                 if os.path.isdir(os.path.join(sensor_path, d))]
    
    # Loop through each date
    for date in sorted(date_dirs):
        directory = os.path.join(sensor_path, date)
        
        print(f"Processing: {sensor}/{date}")
        
        # List to store all detections for this day
        day_detections = []
        
        # Define a custom callback to collect detections
        def on_analyze_complete_custom(recording):
            # Add all detections from this recording to our list
            day_detections.extend(recording.detections)
        
        # Create the batch analyzer for this directory
        batch = DirectoryAnalyzer(
            directory,
            analyzers=[analyzer],
            min_conf=0.1
        )
        
        # Set the callback
        batch.on_analyze_complete = on_analyze_complete_custom
        batch.on_error = on_error
        
        # Process all files in this directory
        batch.process()
        
        # Calculate sum amplitude after processing all files
        day_sum_amplitude = 0.0
        for detection in day_detections:
            amplitude = 1.0  # Assume amplitude = 1.0 for now
            confidence = detection['confidence']
            day_sum_amplitude += amplitude * confidence
        
        # Store the result
        results[sensor][date] = day_sum_amplitude
        
        print(f"  → Sum Amplitude: {day_sum_amplitude:.2f}\n")

        # Print final results
print("\n=== FINAL RESULTS ===")
for sensor in sorted(results.keys()):
    print(f"\n{sensor}:")
    for date in sorted(results[sensor].keys()):
        print(f"  {date}: {results[sensor][date]:.2f}")



        
