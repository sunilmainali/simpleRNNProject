import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment_interactive(model_path='simple_rnn.h5', voc_size=30000, max_len=50):
    # Load trained model
    model = load_model(model_path)

    # Manually define class mapping (adjust based on your training labels!)
    sentiment_map = {
        0: 'IRRELEVANT',
        1: 'NEGATIVE',
        2: 'NEUTRAL',
        3: 'POSITIVE'
    }

    print(" Sentiment Predictor — type 'exit' to quit.")
    while True:
        user_input = input("Enter a review/sentence: ")
        if user_input.lower() == 'exit':
            print("Exiting sentiment predictor.")
            break

        # One-hot encode and pad
        encoded = one_hot(user_input, voc_size)
        padded = pad_sequences([encoded], maxlen=max_len, padding='pre')

        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_class = int(np.argmax(prediction))

        # Use the sentiment map to get the label
        predicted_label = sentiment_map.get(predicted_class, f"CLASS {predicted_class}")

        print(f"\n Predicted Sentiment: **{predicted_label}**\n")

# Run
if __name__ == "__main__":
    predict_sentiment_interactive()
