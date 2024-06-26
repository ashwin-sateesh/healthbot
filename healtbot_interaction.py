# -*- coding: utf-8 -*-
"""Healtbot_Interaction (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1i2s2jALteElbRNlpmEf8xWGZzQsHJS6W
"""

from google.colab import drive
drive.mount('/content/gdrive')

import nltk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.chat.util import Chat, reflections
import pickle
import re
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import nltk
import torch.nn as nn
import torch.nn.functional as F
nltk.download('stopwords')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

path = "/content/gdrive/MyDrive/healthbot/"
file_name = 'med_data_all.pkl'
with open(path + file_name , 'rb') as file:
    med_data = pickle.load(file)

#loading the knowledge graph
path = "/content/gdrive/MyDrive/healthbot/"
file_name = 'knowledge_graph.pkl'
with open(path + file_name , 'rb') as file:
    knowledge_graph = pickle.load(file)

def preprocess(inp):
  preprocessed_texts = []
  stop_words = set(stopwords.words('english'))
  for text in inp:

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #remove stop words
    #words = text.split()
    #filtered_words = [word for word in words if word.lower() not in stop_words]
    #preprocessed_texts.append(' '.join(filtered_words))
    preprocessed_texts.append(text)
  return preprocessed_texts

preprocessed_texts = preprocess(med_data['inputs'])
le = LabelEncoder()
new_labels = le.fit_transform(med_data['labels'])
class_names = le.classes_

label_mapping = {class_name: index for index, class_name in enumerate(class_names)}
print(label_mapping)

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # Multiplying by 2 because we have bidirectional LSTM

    def attention(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_size*2, 1)  # Reshape final_state for attention calculation
        attention_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attention_weights = F.softmax(attention_weights, dim=1).unsqueeze(2)
        attention_output = torch.bmm(lstm_output.permute(0,2,1), soft_attention_weights).squeeze(2)
        return attention_output.to(device)

    def forward(self, x):
        x = x.to(device)
        embedded = self.embedding(x)
        embedded = embedded.to(device)
        lstm_out, (hn, cn) = self.lstm(embedded)
        attention_out = self.attention(lstm_out, hn[-2:])
        output = self.fc(attention_out)
        return output.to(device)

lstm_w_attn_path = '/content/gdrive/MyDrive/healthbot/LSTMwA/lstm_w_attn_full_final.pt'
tokenizer_path = '/content/gdrive/MyDrive/healthbot/LSTMwA/'

if device != 'cuda':
  lstm_w_attn = torch.load(lstm_w_attn_path,map_location=torch.device('cpu'))
else:
  stm_w_attn = torch.load(lstm_w_attn_path)

lstm_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
lstm_w_attn = lstm_w_attn.to(device)
lstm_w_attn.eval()

# Find the maximum length of the sequences
def ic_preprocess(preprocessed_txts, lbls):
  # Tokenize and encode the text data
  input_ids = []
  attention_masks = []
  max_length = 0
  for text in preprocessed_txts:
    encoded_dict = tokenizer(text, add_special_tokens=True, return_tensors='pt')
    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    input_ids.append(input_id)
    attention_masks.append(attention_mask)
    max_length = max(max_length, input_id.shape[1])

# Pad the sequences
  for i in range(len(input_ids)):
    input_ids[i] = torch.cat([input_ids[i], torch.zeros(1, max_length - input_ids[i].shape[1]).long()], dim=1)
    attention_masks[i] = torch.cat([attention_masks[i], torch.zeros(1, max_length - attention_masks[i].shape[1]).long()], dim=1)

  input_ids = torch.cat(input_ids, dim=0).to(device)
  attention_masks = torch.cat(attention_masks, dim=0).to(device)
  if lbls is not None:
    labels = torch.tensor(lbls).to(device)
    return input_ids, attention_masks, labels

  else:
    return input_ids, attention_masks

def preprocess_for_lstm(text):
  text = [text]
  prepro_txts = preprocess(text)
  ip_ids, attn_mask = ic_preprocess(prepro_txts, lbls=None)
  ip_ids = ip_ids.to(device)
  attn_mask = attn_mask.to(device)

  with torch.no_grad():
    outputs = lstm_w_attn(ip_ids)#, attention_mask=attn_mask)
    _, predicts = torch.max(outputs.data, 1)

  inverted_label_dict = {v: k for k, v in label_mapping.items()}

  if isinstance(predicts, torch.Tensor):
    predicts = predicts.cpu().numpy()  # If predicts is a tensor
  predicts = [int(label) for label in predicts]  # Convert to int if they are numpy types or Python tensors

  # Map the predicted labels to string labels
  diseases = [inverted_label_dict[label] for label in predicts]
  return diseases[0]

gpt2_model_path = '/content/gdrive/MyDrive/healthbot/gpt2-new'
# Load your fine-tuned GPT-2 model and tokenizer

tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
model = GPT2LMHeadModel.from_pretrained(gpt2_model_path, output_hidden_states=True)
model.to(device)

def generate_answer(model, tokenizer, prompt, max_length=300):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Adjusting generation parameters
    output = model.generate(
        input_ids,
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        temperature=2.0,  # Adjust for randomness
        top_p=0.9,        # Nucleus sampling
        no_repeat_ngram_size=2,  # Prevent repeating n-grams
        early_stopping=True,
        do_sample=True,
        num_beams=2
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Optionally, remove the prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    greeting = "Here's what I found for you:\n"
    return greeting + generated_text

def generate_prompt(inp_text, disease, knowledge_graph):
    start_prompt = f"For the context of {disease}, please consider the symptom and medicine information below:\n\n"

    # Include symptom and medicine information from the knowledge graph
    symptom_info = "\n".join([f"- {symptom}" for symptom in knowledge_graph[disease]['symptoms']])
    medicine_info = "\n".join([f"- {medicine}" for medicine in knowledge_graph[disease]['medicines']])

    prompt = start_prompt + f"Symptoms for {disease}:\n{symptom_info}\n\nMedicines for {disease}:\n{medicine_info}\n\n" + inp_text

    # Instruct the model to answer the question
    prompt += "\n\nPlease provide an answer to the following question:"
    return prompt

import nltk
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.chat.util import Chat, reflections


#model.to(device)
model.eval()

# Pairs is a list of patterns and responses.
pairs = [
    [
        r"hi|hello|hey",
        ["Hello, I'm HealthBot. How can I assist you today?",]
    ],
    [
        r"what are the symptoms of (.*)",
        ["The symptoms of %1 include...", "Symptoms of %1 can range from...",]
    ],
    [
        r"how to treat (.*)",
        ["Treatment for %1 includes...",]
    ],
    [
        r"what is (.*) used for",
        ["%1 is used to treat...",]
    ],
    [
        r"quit",
        ["Goodbye! If you have more questions in the future, don't hesitate to ask.",]
    ],
]

# This is a simple reflection function that can be used to flip a few pronouns
# e.g. 'I am' becomes 'you are'
reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

# Create your own Chatbot
health_bot = Chat(pairs, reflections)

def health_bot_chat():
    print("HealthBot\n---------")
    print("Hello! I'm HealthBot. I can help you with general health questions.\nType 'quit' to leave the chat.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("HealthBot: Goodbye!")
            break

        disease = preprocess_for_lstm(user_input)
        new_input = generate_prompt(user_input, disease, knowledge_graph)
        # Generate a response using your GPT-2 model
        response = generate_answer(model,tokenizer,new_input)
        print("HealthBot:", response)

health_bot_chat()

health_bot_chat()



