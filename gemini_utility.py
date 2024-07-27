import os
import json

import google.generativeai as genai

# get working directoey 
working_directory = os.path.dirname(os.path.abspath(__file__))

config_file_path = f'{working_directory}/config.json'
config_data = json.load(open(config_file_path))

# loading the api key
GOOGLE_API_KEY = config_data['GOOGLE_API_KEY']

# configuring google.generativeai with API key
genai.configure(api_key=GOOGLE_API_KEY)


# Function to load Gemini-Pro model for ChatBot
def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel('gemini-pro')
    return gemini_pro_model


# Get response from gemini-1.5-flash model - image/text to text
def gemini_flash_response(prompt, image):
    gemini_flash_model = genai.GenerativeModel('gemini-1.5-flash')
    response = gemini_flash_model.generate_content([prompt, image])
    result = response.text
    return result


# get response from embeddings model - text to embeddings
def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list


# get response from Gemini-Pro model - text to text (LLM)
def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel('gemini-pro')
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result