import json
import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# Load FAQ data
with open('faqs.json', 'r') as file:
    faqs = json.load(file)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# Text preprocessing function
def preprocess(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Tokenize text
    tokens = word_tokenize(text)

    # Join tokens back
    return " ".join(tokens)

# Preprocess all FAQ questions
processed_questions = [
    preprocess(q) for q in questions
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert questions into vectors
question_vectors = vectorizer.fit_transform(
    processed_questions
)

print("FAQ Chatbot Started!")
print("Type 'exit' to stop.\n")

while True:

    # User input
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Preprocess user question
    processed_input = preprocess(user_input)

    # Convert user question to vector
    user_vector = vectorizer.transform(
        [processed_input]
    )

    # Calculate similarity
    similarity = cosine_similarity(
        user_vector,
        question_vectors
    )

    # Find best match index
    best_match_index = similarity.argmax()

    # Get best similarity score
    best_score = similarity[0][best_match_index]

    # If similarity is good enough
    if best_score > 0.3:

        response = answers[best_match_index]

    else:

        response = "Sorry, I don't understand."

    print("Chatbot:", response)