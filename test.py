#before running use pip install birdnetlib and pprint (also maybe .batch and .analyzer separately. I can't remember)

from datetime import datetime
from birdnetlib import Recording
from birdnetlib.batch import DirectoryAnalyzer
from birdnetlib.analyzer import Analyzer
from pprint import pprint
custom_list_path = "species_list.txt"  # See example file for formatting.
#change your path of course. I will zip up some audio files.
data_path = "D:/Shelburne/May/Sensor 5/20250505"
data_path = "./ShelburneSubset/Sensor 1/20250505"


def on_analyze_complete(recording: Recording):
    #recording.extract_embeddings()
    print(f"Completed analysis for {recording.file_path}")
    #pprint(recording.detections)
    #recording.extract_detections_as_audio(directory="./exports",format="wav")
    #print("waves "+waves)
    #for w in waves:
    #    print(w)

def on_error(recording,error: Exception):
    print(f"Error occurred: {error}")
    print(f"While processing file: {recording.file_path}")   
    
analyzer = Analyzer(custom_species_list_path=custom_list_path)
batch = DirectoryAnalyzer(
    data_path,
    analyzers=[analyzer],
    min_conf=0.1
)

batch.on_analyze_complete = on_analyze_complete
batch.on_error = on_error
batch.extract_detections_as_spectrograms = False
batch.extract_detections_as_audio = True
batch.extract_detections_directory = "./exports"
batch.process()
# recording = Recording(
#     analyzer,
#     "./test_audio/test3.wav",
#     min_conf=0.2,

#     date=datetime(year=2025, month=5, day=5),
#     return_all_detections=False
# )
#pprint(recording)

#recording.analyze()
#recording.extract_detections_as_audio(directory="./exports",format="wav")
#the recording.detections will store the confidence 
#recording.extract_detections_as_spectrogram(directory="C:/Users/greeneks/OneDrive - Thomas College/Documents-PC/Birds/Grassland/Data/Training/Extracted")
#pprint(recording.detections)