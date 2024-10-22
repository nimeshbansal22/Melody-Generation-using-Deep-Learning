
# Melody Generation using Deep Learning

## Project Overview
This project aims to generate music melodies using deep learning techniques. The goal is to train a model that can create coherent musical sequences based on existing datasets of music. The generated melodies can be used for creative purposes, such as composing new pieces or enhancing musical production workflows. The project utilizes TensorFlow, Keras, and other related tools to preprocess data, train models, and generate new music based on given input sequences.

## Features
- **Data Preprocessing:** Load and preprocess music files from the dataset to prepare them for training. The preprocessing includes filtering files with acceptable note durations, extracting musical features, and preparing sequences for training.
- **Parallel Data Loading:** The project uses multi-processing to efficiently load and parse a large number of music files from the dataset, significantly reducing processing time.
- **Deep Learning Model Training:** The model is trained using a recurrent neural network (RNN) architecture to understand musical patterns and generate new melodies. The training process is customizable with various parameters, such as number of epochs and learning rate.
- **Melody Generation:** The project provides functionality for generating new melodies given a seed sequence. Users can specify parameters like temperature to influence the creativity and randomness of the generated melody.
- **MIDI File Export:** Generated melodies can be saved as MIDI files for further use in music production or playback.

## File Structure
- **`dataset` folder:** Directory containing the preprocessed music dataset.
- **`all_dataset` file:** A consolidated file containing all sequences for training.
- **`mapping.json` file:** A JSON file mapping musical notes to integers used for training the model.
- **`model.h5` file:** The trained model that is used for melody generation.

## Installation
To run this project, you need to install the following dependencies:

```bash
pip install tensorflow keras music21 numpy
```

Make sure to have Python 3.6+ and other necessary libraries installed.

## Usage
1. **Load and Preprocess Data:** Use the `load_songs()` function to load the music dataset. The function utilizes parallel processing to efficiently load `.krn` files from the given directory.
2. **Train the Model:** Run the script to preprocess the dataset, convert it into sequences, generate a mapping, and train the model. Training parameters like batch size and epochs can be adjusted to improve results.
3. **Generate Melody:** Use the `MelodyGeneration` class to generate a melody. You can provide a seed sequence, specify the number of steps for generation, and adjust temperature to control the level of randomness in the generated output.
4. **Save Generated Melody:** Save the generated melody as a MIDI file using the `save_melody()` function for playback or further use.

### Example Command
Run the following command to start the entire process:

```python
python main.py
```

This script will:
- Load and preprocess the dataset.
- Train the model on preprocessed sequences.
- Generate a melody using a seed sequence and save it as a MIDI file.

## Key Parameters
- **Dataset Path (`dataset_path`)**: Path to the folder containing the music dataset files.
- **Acceptable Durations (`ACCEPTABLE_DURATIONS`)**: A list of allowed note durations for filtering songs during preprocessing.
- **Model Training (`EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`)**: Parameters to control the training process of the model.
- **Melody Generation (`SEQUENCES`, `OUTPUT_UNITS`)**: Parameters that define the sequence length for melody generation and the number of output units for the model.

## Requirements
- **Python 3.6+**
- **TensorFlow**
- **Keras**
- **Music21**: Used for parsing music files and generating MIDI files.
- **NumPy**
- **Multiprocessing**: Used for efficient data loading.

## Example Usage
An example melody generation process is initiated with a seed sequence:

```python
mg = MelodyGeneration()
seed = "67 _ 67 _ 65 67 _ _ 65 64 _ 64 _ 64 _ _"
melody = mg.melody_generate(seed, 500, SEQUENCES, 0.8)
print(melody)
mg.save_melody(melody)
```

## Acknowledgements
- This project uses musical datasets available in `.krn` format, specifically a dataset related to classical music (`erk` dataset).
- The project utilizes `music21` for music parsing and TensorFlow/Keras for model building and training.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as per the license agreement.

## Contact
For further questions or collaboration, please reach out at [nimeshbansal3390@gmail.com].
