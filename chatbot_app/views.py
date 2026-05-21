from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings

import json
import string
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# FAQ file path
faq_path = os.path.join(settings.BASE_DIR, 'faqs.json')

with open(faq_path, 'r') as file:
    faqs = json.load(file)

questions = [faq['question'] for faq in faqs]
answers = [faq['answer'] for faq in faqs]


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)


def home(request):
    return render(request, 'index.html')


def chat(request):

    if request.method == 'POST':

        try:
            data = json.loads(request.body)
            user_input = data.get('message', '')

            processed_input = preprocess(user_input)

            user_vector = vectorizer.transform([processed_input])

            similarity = cosine_similarity(user_vector, question_vectors)

            best_match_index = similarity.argmax()
            best_score = similarity[0][best_match_index]

            if best_score > 0.2:
                response = answers[best_match_index]
            else:
                response = "Sorry, I don't understand."

            return JsonResponse({'response': response})

        except Exception as e:
            print("ERROR:", e)
            return JsonResponse({'response': 'Backend Error'})

    return JsonResponse({'response': 'Invalid Request'})