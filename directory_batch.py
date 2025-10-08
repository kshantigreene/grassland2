
from birdnetlib.batch import DirectoryAnalyzer
from birdnetlib.analyzer import Analyzer
from datetime import datetime
from pprint import pprint
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
directory = "./ShelburneSubset/Sensor2/20250506"

batch = DirectoryAnalyzer(
    directory,
    analyzers=[analyzer], #use the analyzer with custom list path
    #lon=-120.7463, 
    #lat=35.4244,+1
    #date=datetime(year=2022, month=5, day=10),
    min_conf=0.1 #minimum confidence. Keep this pretty low
    
)

batch.on_analyze_complete = on_analyze_complete
batch.on_error = on_error
batch.process()
