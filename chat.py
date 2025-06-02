import random
import json
import torch
from neural_net import NeuralNet
from flask_server.university.nlp_utils import bag_of_words, tokenize

# Load intents with UTF-8 encoding
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FILE = "data.pth"
data = torch.load(FILE)

# Load model parameters
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()
model.to(device)


def chatbot_response(sentence):
    tokenized_sentence = tokenize(sentence.lower())
    X = bag_of_words(tokenized_sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    probabilities = torch.softmax(output, dim=1)
    max_prob, predicted = torch.max(probabilities, dim=1)
    tag = tags[predicted.item()]
    prob = max_prob.item()

    print(f"Input: {sentence}")
    print(f"Predicted tag: {tag}, Confidence: {prob:.4f}")

    if prob > 0.4:  # Increased threshold
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                # Normalize response to always be a dictionary
                if isinstance(response, str):
                    response_dict = {"text": response, "media": [], "link": ""}
                else:
                    response_dict = (
                        response  # Already a dictionary with text, media, link
                    )
                return (response_dict, tag)
    # Default response if confidence is too low
    return (
        {"text": "Please modify your query little bit..", "media": [], "link": ""},
        "unknown",
    )


# Test the function
print(chatbot_response("hi"))
