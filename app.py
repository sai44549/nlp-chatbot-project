import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 📚 Knowledge Base (Edit here)
# -----------------------------
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

# -----------------------------
# 🧠 NLP Logic
# -----------------------------
def get_response(user_input):
    # Combine questions + user input
    texts = questions + [user_input]

    # Convert text to vectors (numbers)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)

    # Compare similarity
    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    # Get best match index
    best_match_index = similarity.argmax()

    # Return corresponding answer
    return answers[best_match_index]


# -----------------------------
# 🌐 Streamlit UI
# -----------------------------
st.title("💬 NLP Chatbot")

# Store chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Input box
user_input = st.text_input("You:")

# When user sends message
if user_input:
    bot_reply = get_response(user_input)

    # Save conversation
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", bot_reply))

# Display chat history
for sender, message in st.session_state.chat:
    st.write(f"**{sender}:** {message}")