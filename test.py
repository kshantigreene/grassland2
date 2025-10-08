#before running use pip install birdnetlib and pprint (also maybe .batch and .analyzer separately. I can't remember)

from datetime import datetime
from birdnetlib import Recording
from birdnetlib.batch import DirectoryAnalyzer
from birdnetlib.analyzer import Analyzer
from pprint import pprint
custom_list_path = "species_list.txt"  # See example file for formatting.
#change your path of course. I will zip up some audio files.
data_path = "D:/Shelburne/May/Sensor 5/20250505"


def on_analyze_complete(recording: Recording):
    print(f"Completed analysis for {recording.file_path}")
    pprint(recording.detections)

def on_error(error: Exception):
    print(f"Error occurred: {error}")   
    
analyzer = Analyzer(custom_species_list_path=custom_list_path)
batch = DirectoryAnalyzer(
    data_path,
    analyzers=[analyzer],
    min_conf=0.1
)

batch.on_analyze_complete = on_analyze_complete
batch.on_error = on_error
batch.extract_detections_as_spectrograms = False
batch.extract_detections_directory = "C:/Users/greeneks/OneDrive - Thomas College/Documents-PC/Birds/Grassland/Data/Training/Extracted"
#batch.process()
recording = Recording(
    analyzer,
    "test_audio/test3.wav",
    min_conf=0.4,

    date=datetime(year=2025, month=5, day=5),
    return_all_detections=False
)
recording.analyze()
#the recording.detections will store the confidence 
#pprint(recording.detections)
# Extract to spectrograms
#recording.extract_detections_as_spectrogram(directory="C:/Users/greeneks/OneDrive - Thomas College/Documents-PC/Birds/Grassland/Data/Training/Extracted")

pprint(recording.detections)
