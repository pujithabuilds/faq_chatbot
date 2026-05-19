from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

import json
import nltk
import string
import os

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download tokenizer
nltk.download('punkt')

# Correct FAQ path
faq_path = os.path.join(settings.BASE_DIR, 'faqs.json')

# Load FAQs
with open(faq_path, 'r') as file:
    faqs = json.load(file)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]

# Preprocess text
def preprocess(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    tokens = word_tokenize(text)

    return " ".join(tokens)

# Process all questions
processed_questions = [
    preprocess(q) for q in questions
]

# Create vectors
vectorizer = TfidfVectorizer()

question_vectors = vectorizer.fit_transform(
    processed_questions
)

# Home page
def home(request):

    return render(request, 'index.html')

# Chat API
def chat(request):

    if request.method == 'POST':

        data = json.loads(request.body)

        user_input = data['message']

        print("User Input:", user_input)

        processed_input = preprocess(user_input)

        user_vector = vectorizer.transform(
            [processed_input]
        )

        similarity = cosine_similarity(
            user_vector,
            question_vectors
        )

        best_match_index = similarity.argmax()

        best_score = similarity[0][best_match_index]

        print("Best Score:", best_score)
        print("Best Match:", questions[best_match_index])

        if best_score > 0.2:

            response = answers[best_match_index]

        else:

            response = "Sorry, I don't understand."

        return JsonResponse({
            'response': response
        })