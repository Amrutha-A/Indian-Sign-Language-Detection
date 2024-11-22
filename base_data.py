import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Read the Excel file to get the label mapping
excel_file = r"D:\Silcosys\lsa preprocessed\meta.xlsx"
df = pd.read_excel(excel_file)

# Create a mapping of IDs to names
label_map = {str(row['ID']).zfill(3): row['Name'] for _, row in df.iterrows()}

# Directory paths
video_directory = r"D:\Silcosys\lsa preprocessed\all"
output_data_path = r"D:\Silcosys\lsa preprocessed\distance"

# Ensure output directory exists
if not os.path.exists(output_data_path):
    os.makedirs(output_data_path)

# MediaPipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Extract keypoints (reuse your logic)
def extract_keypoints(results):
    # Extract coordinates
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    mouth_left_x, mouth_left_y = pose[9*4], pose[9*4 + 1]
    shoulder_left_x, shoulder_left_y = pose[11*4], pose[11*4 + 1]
    shoulder_right_x, shoulder_right_y = pose[12*4], pose[12*4 + 1]
    elbow_left_x, elbow_left_y = pose[13*4], pose[13*4 + 1]
    elbow_right_x, elbow_right_y = pose[14*4], pose[14*4 + 1]
    nose_x, nose_y = pose[0], pose[1]

    left_thumb_x, left_thumb_y = lh[4*3], lh[4*3 + 1]
    right_thumb_x, right_thumb_y = rh[4*3], rh[4*3 + 1]
    right_index_x, right_index_y = rh[8*3], rh[8*3 + 1]
    left_index_x, left_index_y = lh[8*3], lh[8*3 + 1]
    right_middle_x, right_middle_y = rh[12*3], rh[12*3 + 1]
    left_middle_x, left_middle_y = lh[12*3], lh[12*3 + 1]
    right_ring_x, right_ring_y = rh[16*3], rh[16*3 + 1]
    left_ring_x, left_ring_y = lh[16*3], lh[16*3 + 1]
    right_pinky_x, right_pinky_y = rh[20*3], rh[20*3 + 1]
    left_pinky_x, left_pinky_y = lh[20*3], lh[20*3 + 1]

    right_hand_x, right_hand_y = rh[0], rh[1]
    left_hand_x, left_hand_y = lh[0], lh[1]

    distances = np.array([
        np.sqrt((mouth_left_x - left_thumb_x)**2 + (mouth_left_y - left_thumb_y)**2),
        np.sqrt((mouth_left_x - right_thumb_x)**2 + (mouth_left_y - right_thumb_y)**2),
        np.sqrt((left_thumb_x - right_thumb_x)**2 + (left_thumb_y - right_thumb_y)**2),
        np.sqrt((right_ring_x - left_ring_x)**2 + (right_ring_y - left_ring_y)**2),
        np.sqrt((right_middle_x - left_middle_x)**2 + (right_middle_y - left_middle_y)**2),
        np.sqrt((right_index_x - left_index_x)**2 + (right_index_y - left_index_y)**2),
        np.sqrt((right_pinky_x - left_pinky_x)**2 + (right_pinky_y - left_pinky_y)**2),
        np.sqrt((right_hand_x - left_hand_x)**2 + (right_hand_y - left_hand_y)**2),
        np.sqrt((right_hand_x - nose_x)**2 + (right_hand_y - nose_y)**2),
        np.sqrt((left_hand_x - nose_x)**2 + (left_hand_y - nose_y)**2),
        np.sqrt((nose_x - right_index_x)**2 + (nose_y - right_index_y)**2),
        np.sqrt((nose_x - left_index_x)**2 + (nose_y - left_index_y)**2),
        np.sqrt((nose_x - right_middle_x)**2 + (nose_y - right_middle_y)**2),
        np.sqrt((nose_x - left_middle_x)**2 + (nose_y - left_middle_y)**2),
        np.sqrt((shoulder_left_x - left_hand_x)**2 + (shoulder_left_y - left_hand_y)**2),
        np.sqrt((shoulder_right_x - right_hand_x)**2 + (shoulder_right_y - right_hand_y)**2),
        np.sqrt((elbow_right_x - right_hand_x)**2 + (elbow_right_y - right_hand_y)**2),
        np.sqrt((elbow_left_x - left_hand_x)**2 + (elbow_left_y - left_hand_y)**2),
        np.sqrt((elbow_left_x - nose_x)**2 + (elbow_left_y - nose_y)**2),
        np.sqrt((elbow_right_x - nose_x)**2 + (elbow_right_y - nose_y)**2)
    ])

    return np.concatenate([pose, lh, rh, distances])

# Extract landmarks from a single video
def extract_landmarks_from_video(video_path, holistic_model):
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_keypoints(results)
        frame_landmarks.append(keypoints)

    cap.release()
    return np.array(frame_landmarks)

# Process videos and save landmarks
processed_counts = {}  # To track processed video counts for each sign
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for video_file in sorted(os.listdir(video_directory)):  # Sort to ensure consistent order
        video_path = os.path.join(video_directory, video_file)

        # Extract IDs from the video filename (e.g., "001_001_001.mp4")
        file_parts = video_file.split("_")
        word_id, person_id, video_id = file_parts[0], file_parts[1], os.path.splitext(file_parts[2])[0]

        # Check if this sign (word_id) has already reached the limit of 5 videos
        if word_id not in processed_counts:
            processed_counts[word_id] = 0

        if processed_counts[word_id] >= 5:
            continue

        # Get the label name
        label_name = label_map.get(word_id)
        if not label_name:
            print(f"Label not found for ID {word_id}")
            continue

        # Extract landmarks
        landmarks = extract_landmarks_from_video(video_path, holistic)

        # Save landmarks with a clear naming convention
        npy_filename = f"{label_name}_Person{person_id}_Video{video_id}.npy"
        output_label_path = os.path.join(output_data_path, label_name)
        if not os.path.exists(output_label_path):
            os.makedirs(output_label_path)
        np.save(os.path.join(output_label_path, npy_filename), landmarks)

        # Increment the processed count for this sign
        processed_counts[word_id] += 1

        print(f"Saved landmarks for {video_file} as {npy_filename}")
