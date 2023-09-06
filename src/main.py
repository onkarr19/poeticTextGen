import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers.legacy import RMSprop
import os

filepath = '../data/raw/shakespeare.txt'

# import tensorflow as tf
# filepath = tf.keras.utils.get_file('shakespeare.txt', 
#                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

data = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = data[:]

# Create a sorted list of unique characters in the 'text'.
characters = sorted(set(text))
# print(type(characters), characters)


# Create a dictionary 'char_to_index' to map characters to their indices.
char_to_index = {c : i for i, c in enumerate(characters)}

# Create a dictionary 'index_to_char' to map indices back to characters.
index_to_char = {i : c for i, c in enumerate(characters)}

# Define the length of sequences to extract from the 'text'.
SEQUENCE_LENGTH = 60

# Define the step size for moving through the 'text' to create sequences.
STEP_SIZE = 4

# # Initialize empty lists to store sequences and their corresponding next characters.
# sequences = []
# next_characters = []

# # Create sequences of characters by sliding a window of size 'SEQUENCE_LENGTH' with a 
# # step size of 'STEP_SIZE' through the 'text'.
# for i in range(0, len(text)-SEQUENCE_LENGTH, STEP_SIZE):
#     sequences.append(text[i: i+SEQUENCE_LENGTH])
#     next_characters.append(text[i+SEQUENCE_LENGTH])


# # Initialize a numpy array 'x' to store the one-hot encoded sequences.
# x = np.zeros((len(sequences), SEQUENCE_LENGTH, len(characters)), dtype=np.bool_)

# # Initialize a numpy array 'y' to store the one-hot encoded next characters.
# y = np.zeros((len(sequences), len(characters)), dtype=np.bool_)

# # Loop through the sequences and one-hot encode the characters.
# for i, sequence in enumerate(sequences):
#     for j, char in enumerate(sequence):
#         # Set the corresponding character's index to 1 in the 'x' array.
#         x[i, j, char_to_index[char]] = 1
#     # Set the corresponding next character's index to 1 in the 'y' array.
#     y[i, char_to_index[next_characters[i]]] = 1



# # Create a Sequential model for a character-level text generation task.

# # Add an LSTM layer with 128 units, taking input sequences of length SEQUENCE_LENGTH 
# # and with input features equal to the number of unique characters in the dataset.
# model = Sequential()
# model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))

# # Add a Dense layer with as many units as there are unique characters in the dataset.
# model.add(Dense(len(characters)))

# # Apply a Softmax activation function to the Dense layer's output to obtain character probabilities.
# model.add(Activation('Softmax'))

# # Compile the model with categorical cross-entropy loss and RMSprop optimizer with a learning rate of 0.01.
# model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))

# # Train the model using input data 'x' and target data 'y', with a batch size of 256 and for 4 epochs.
# model.fit(x, y, batch_size=256, epochs=4)

# # Save the trained model to the specified directory for future use.
# # model.save(os.path.join('../output', 'poetic_text.model'))


# Load the saved model for further use.
model = load_model('../output/poetic_text.model')

def sample(preds, temperature=1.0):
    """
    Generate a sample from a probability distribution with optional temperature scaling.

    Parameters:
    - preds (list): A list of probabilities representing a probability distribution.
    - temperature (float): A parameter for scaling the sampling randomness.
                           Higher values make the sampling more random (diverse),
                           while lower values make it more deterministic (greedy).

    Returns:
    - int: The index of the sampled element from the probability distribution.
    """

    # Ensure the input probabilities are in the correct data type
    preds = np.asarray(preds).astype('float64')

    # Apply temperature scaling to the log probabilities
    preds = np.log(preds) / temperature

    # Exponentiate the scaled probabilities
    exp_preds = np.exp(preds)

    # Normalize the probabilities to sum to 1
    preds = exp_preds / np.sum(exp_preds)

    # Generate a random sample from the multinomial distribution
    probas = np.random.multinomial(1, preds, 1)

    # Return the index of the sampled element
    return np.argmax(probas)



import numpy as np

def generate_text(previous, length, temperature=0.1):
    # Initialize the generated text with the previous text
    sentence = previous[:]
    generated = ''
    generated += sentence

    # Generate 'length' characters
    for i in range(length):
        # Create an input tensor for the model
        x_predictions = np.zeros((1, SEQUENCE_LENGTH, len(characters)))

        # Encode the characters in the 'sentence' into a one-hot representation
        for t, char in enumerate(sentence):
            x_predictions[0, t, char_to_index[char]] = 1

        # Make predictions using the model
        predictions = model.predict(x_predictions, verbose=0)[0]

        # Sample the next character based on the predicted probabilities and temperature
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        # Append the generated character to the output text
        generated += next_character

        # Update the 'sentence' by removing the first character and adding the new character
        sentence = sentence[1:] + next_character

    # Return the generated text
    return generated


# Playground

previous_sentence = text[10000: 10000 + SEQUENCE_LENGTH]
print(previous_sentence)
for i in range(1,10):
    print('-'*50, i/10, '-'*50)
    print(generate_text(previous_sentence, length=300, temperature=i/10))
