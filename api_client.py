"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""   
    # "response_mime_type": "text/plain",

from config import API_KEY
import google.generativeai as genai
genai.configure(api_key= API_KEY)
generation_config = {
#   "temperature": 1,
#   "top_p": 0.95,
#   "top_k": 64,
#   "max_output_tokens": 2048,
     "temperature": 0.9,
    "top_p": 1,
    "top_k": 0,
    # "max_output_tokens": 2048,
    "max_output_tokens": 1024,  # Giới hạn độ dài của câu trả lời
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.0-pro",
#   model_name="gemini-1.5-flash",
  safety_settings=safety_settings,
  generation_config=generation_config,
)
# Create the model once and reuse it
# model = genai.GenerativeModel(
#     model_name="gemini-1.0-pro",
#     safety_settings=safety_settings,
#     generation_config=generation_config,
# )

# Function to start a chat session and send a message, returning the response
def start_chat_and_send_message(message):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(message)
    return response.text, chat_session.history


