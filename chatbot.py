import nltk
from nltk.chat.util import Chat, reflections

pairs = [
    ["hi|hello|hey", ["Hello! How can I help you today?"]],
    ["what is your name ?", ["I am an NLP chatbot created by Sai."]],
    ["how are you ?", ["I'm doing great! Thanks for asking."]],
    ["bye|exit", ["Goodbye! Have a nice day!"]],
]

chat = Chat(pairs, reflections)

print("Chatbot is running... (type 'bye' to exit)")
chat.converse()