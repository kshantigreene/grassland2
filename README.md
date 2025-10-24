# Grassland Bird Audio Analysis

A bioacoustics analysis toolkit for detecting and analyzing grassland bird species from audio recordings, with a focus on Savannah Sparrows and Bobolinks. This project combines BirdNET-based species detection with custom convolutional neural networks (CNNs) for acoustic analysis.

## Overview

This project provides tools for:
- Automated bird species detection using BirdNET
- Batch processing of hierarchical audio data (sensor → date → recordings)
- Audio amplitude analysis and confidence-weighted metrics
- Spectrogram generation from audio files
- Custom CNN training for bird call classification
- Species presence quantification across multiple sensors and time periods

## Project Structure

```
grassland2/
├── directory_batch.py       # Main batch processing script for BirdNET analysis
├── bird_class_clean.ipynb   # CNN model training and evaluation
├── bird_class_backup.ipynb  # Backup of classification notebook
├── cnn.ipynb               # TensorFlow CNN tutorial/template
├── species_list.txt        # Target species for detection
└── test.py                 # Test/utility scripts
```

## Features

### 1. Batch Audio Processing (`directory_batch.py`)
- **Hierarchical Directory Processing**: Automatically processes nested directory structures (Sensor/Date/Recordings)
- **Species Filtering**: Focuses analysis on target species (Savannah Sparrow, Bobolink)
- **Amplitude Metrics**: Calculates sum amplitudes weighted by detection confidence
- **Aggregated Results**: Produces per-sensor, per-day summary statistics

**Data Structure Expected:**
```
ShelburneSubset/
├── Sensor1/
│   ├── 2024-06-01/
│   │   ├── recording_001.wav
│   │   └── recording_002.wav
│   └── 2024-06-02/
│       └── recording_001.wav
└── Sensor2/
    └── 2024-06-01/
        └── recording_001.wav
```

### 2. CNN Classification (`bird_class_clean.ipynb`)
- **Spectrogram Generation**: Converts audio files to frequency spectrograms (2000-10000 Hz)
- **Custom CNN Architecture**: Multi-layer convolutional neural network for bird call classification
- **Training/Test Split**: Automated dataset splitting and evaluation
- **Audio Splicing**: Segments long recordings into fixed-length samples
- **Amplitude Filtering**: Removes low-amplitude segments to improve training quality

## Dependencies

```bash
pip install birdnetlib
pip install librosa
pip install tensorflow
pip install keras
pip install scikit-image
pip install numpy
pip install matplotlib
pip install scipy
```

## Usage

### Running BirdNET Batch Analysis

```python
python directory_batch.py
```

**Configuration:**
- Modify `base_directory` to point to your audio data
- Adjust `min_conf` threshold (default: 0.1) for detection sensitivity
- Edit `species_list.txt` to target different species

**Output:**
```
=== Processing All Sensor-Day Combinations ===

Processing: Sensor1/2024-06-01
  → Sum Amplitude: 15.43

Processing: Sensor1/2024-06-02
  → Sum Amplitude: 8.27

=== FINAL RESULTS ===
Sensor1:
  2024-06-01: 15.43
  2024-06-02: 8.27
```

### Training a CNN Model

Open `bird_class_clean.ipynb` in Jupyter and follow the notebook cells to:

1. **Generate Spectrograms** from audio files
2. **Load Training Data** from labeled directories
3. **Train the Model** with your custom dataset
4. **Evaluate Performance** on test data
5. **Save the Model** for future use

**Key Parameters:**
- `sampling_rate`: 44100 Hz (default)
- `segment_length`: 44100 samples (~1 second)
- `min_freq`: 2000 Hz
- `max_freq`: 10000 Hz
- `minAmp`: 0.01 (amplitude threshold)

## Target Species

Currently configured to detect:
- **Savannah Sparrow** (*Passerculus sandwichensis*)
- **Bobolink** (*Dolichonyx oryzivorus*)

Additional species can be added by editing `species_list.txt` using the format:
```
Scientific_name_Common Name
```

## Algorithm Details

### Amplitude-Weighted Detection

For each sensor-day combination, the system calculates:

```
sum_amplitude = Σ (amplitude × confidence)
```

Where:
- `amplitude`: Currently set to 1.0 (placeholder for future implementation)
- `confidence`: BirdNET detection confidence (0.0-1.0)

This metric quantifies both the presence and certainty of species detections across time periods.

### Spectrogram Processing

Audio is converted to spectrograms using:
- **Stride**: 10ms
- **Window**: 20ms (Hanning window)
- **FFT**: Real FFT with frequency filtering
- **Frequency Range**: 2000-10000 Hz (optimized for small bird vocalizations)

## Results Storage

The script stores results in a nested dictionary structure:
```python
{
  "Sensor1": {
    "2024-06-01": 15.43,
    "2024-06-02": 8.27
  },
  "Sensor2": {
    "2024-06-01": 12.56
  }
}
```

This format facilitates temporal analysis, sensor comparison, and statistical modeling.

## Future Enhancements

- [ ] Implement true amplitude extraction from audio files
- [ ] Add visualization dashboards for temporal patterns
- [ ] Export results to CSV/JSON for further analysis
- [ ] Integrate GPS coordinates for spatial mapping
- [ ] Multi-species simultaneous tracking
- [ ] Real-time processing capabilities

## Research Applications

This toolkit is designed for ecological research involving:
- Avian population monitoring
- Habitat quality assessment
- Breeding season phenology studies
- Acoustic niche analysis
- Long-term biodiversity tracking

## Contributing

Contributions are welcome! Areas for improvement:
- Additional species models
- Enhanced amplitude detection algorithms
- Performance optimization for large datasets
- Automated report generation

## License

This project uses BirdNET, which is available under specific licensing terms. Please review BirdNET's license before commercial use.

## References

- BirdNET: [Cornell Lab of Ornithology](https://birdnet.cornell.edu/)
- Audio processing based on librosa library
- CNN architecture inspired by TensorFlow image classification tutorials

## Contact

For questions about this analysis pipeline, please open an issue in this repository.

---

**Note**: This is a research tool. Detection accuracy depends on recording quality, background noise, and species-specific vocalization patterns. Always validate automated detections with manual review when possible.

## Authors: Kshanti Greene and James Gyampoh, Thomas College
