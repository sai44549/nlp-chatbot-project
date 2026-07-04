import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# download once (first time only)
nltk.download('punkt')

# Knowledge base (questions + answers)
questions = [
    "hello",
    "hi",
    "what is your name",
    "what is artificial intelligence",
    "what is machine learning"
]

answers = [
    "Hello! How can I help you?",
    "Hi there!",
    "I am your NLP chatbot.",
    "Artificial Intelligence is the simulation of human intelligence.",
    "Machine learning is a part of AI."
]

# Function to get best response
def get_response(user_input):
    all_text = questions + [user_input]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_text)

    similarity = cosine_similarity(vectors[-1], vectors[:-1])
    index = similarity.argmax()

    return answers[index]

print("NLP Chatbot running... (type 'bye' to exit)")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print("Bot:", response)