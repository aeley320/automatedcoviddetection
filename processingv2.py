import os
import pandas as pd
import pygame


# Paths
input_folder = "my_data/"  # Replace with the path to your audio files
output_file = "labeled_audio_data.csv"  # CSV file to save labels

# Initialize DataFrame to save labels
if os.path.exists(output_file):
    # Load existing labels if the file exists
    labeled_data = pd.read_csv(output_file)
else:
    labeled_data = pd.DataFrame(columns=["Filename", "Label"])

# Initialize pygame mixer
pygame.mixer.init()

# Play audio and get label input


def play_audio_with_labeling(file_path):
    """
    Play the audio file using pygame and allow user to label it.
    """
    try:
        while True:  # Allow replay functionality
            print(f"Playing: {file_path}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()

            # Wait for user input
            label = input(
                "Enter label (e.g., 1 for cough, 0 for non-cough), 'r' to replay, or 's' to skip: ")
            pygame.mixer.music.stop()  # Stop playback

            if label == "s":  # Skip the file
                print("Skipped!")
                return "skip"
            elif label == "r":  # Replay the current audio
                print("Replaying the audio...")
                continue  # Restart the loop to replay
            elif label.isdigit():  # Validate numeric input
                return int(label)
            else:
                print(
                    "Invalid input. Please enter a numeric label, 'r' to replay, or 's' to skip.")
    except Exception as e:
        print(f"Error playing {file_path}: {e}")
        return None

# Process all audio files in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".wav") and file_name not in labeled_data["Filename"].values:
        file_path = os.path.join(input_folder, file_name)

        # Play audio and get label
        label = play_audio_with_labeling(file_path)
        if label == "skip":
            continue  # Skip this file without saving a label

        if label is not None:
            # Append the label to the DataFrame
            labeled_data = pd.concat([labeled_data, pd.DataFrame(
                {"Filename": [file_name], "Label": [label]})], ignore_index=True)
            # Save progress after each labeling
            labeled_data.to_csv(output_file, index=False)
            print(f"Labeled {file_name} as {label} and saved to {output_file}")

print("Labeling complete!")
pygame.mixer.quit()  # Clean up pygame mixer
