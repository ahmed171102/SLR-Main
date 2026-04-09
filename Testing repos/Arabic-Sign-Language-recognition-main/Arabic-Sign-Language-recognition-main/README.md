# Arabic Sign Language Recognition

A project focused on enhancing the recognition accuracy of Arabic sign language gestures using efficient Deep learning models.

## Description

In the domain of Sign Language Recognition (SLR), different models have made significant progress in recent years. However, many models face challenges such as accurately capturing complex gestures, handling a wide range of classes, and meeting demanding computational requirements for real-time predictions, resulting in slow processing speeds.

This project demonstrates two approaches based on pose estimation of landmarks to effectively handle both word-level and character-level gestures. The primary objective is to achieve high recognition accuracy across a large dataset, particularly for complex signs, using efficient classifiers and less computational models, ensuring real-time performance.

The proposed models consist of:
- **Multi-layer Perceptron (MLP) classifier** for recognizing static signs, such as individual(static) characters.![MLP architecture](https://github.com/user-attachments/assets/7140f5d0-b74d-41fa-b804-b35abb5ffa1b)

- **Long Short-Term Memory (LSTM) network** for recognizing dynamic signs, such as complex(dynamic) gestures that involve motion.
  ![LSTM in real time](https://github.com/user-attachments/assets/fb02f84a-f30a-465c-854b-f47adfddb022)


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Acknowledgments](#acknowledgments)

## Installation

To set up this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/arabic-sign-language-recognition.git
   cd arabic-sign-language-recognition
## Usage 
Users can interact with this project by any IDE for Python i recommend Visual studio Code and install all libraries in requirements file [requirements.txt file](https://github.com/abdelrhmanmousa/Arabic-Sign-Language-recognition/blob/main/requirements.txt)

## Features
Here's a detailed explanation of sign to text functionality within the software:
+ Gesture detection: The software utilizes computer vision techniques and
  
   MediaPipe to detect and analyze Arabic sign language data.

+ Gesture Recognition: Gestures are detected using deep learning algorithms
   such as LSTMSs and MLP.

+ Translation: The software recognizes gestures and maps them to Arabic text
  
   representations. Each Arabic Sign Language sign corresponds to a specific word,

   phrase, or concept in written or typed Arabic text.

+ Real-time Feedback: The sign to text conversion process is real-time, allowing
 
   for instant translation of sign language gestures into text.

+ Contextual Understanding: The software considers individual signs and
  
   conversation context to improve translation accuracy and relevance. It takes into

   account factors such as the sequence of gestures for a more comprehensive

   understanding of communication.

+ Adaptation and Learning: The software may use adaptive learning
 
   mechanisms to enhance recognition accuracy over time, incorporating feedback

   loops for user correction and refinement of algorithms and models.

+ Output Options: The translated text output can be displayed in various formats,
  
   including on-screen text or as input to other applications for further processing

   or communication

