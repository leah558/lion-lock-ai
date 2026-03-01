from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    
except Exception:
    model = None

app = Flask(__name__)
app.secret_key = os.urandom(24)


def compare_styles_tfidf(existing_texts, new_text):
    if not existing_texts or not new_text:
        return None

    try:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
        all_texts = existing_texts + [new_text]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        return float(similarity_scores.mean())
    except Exception:
        return None

def compare_styles_gemini(existing_texts, new_text):
    if not existing_texts or not new_text or model is None:
        return None, None

    prompt = (
        "You are a writing style comparison expert.\n\n"
        "Existing emails:\n"
        + "\n---\n".join(existing_texts)
        + "\n\nNew email:\n"
        + new_text +
        "\n\nRate similarity from 0.0 to 1.0 on the first line only."
        " Then explain briefly."
    )

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        first_line = text.splitlines()[0]
        match = re.search(r"([01](?:\.\d+)?)", first_line)
        score = float(match.group(1)) if match else None

        return score, text

    except Exception as e:
        return None, str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    authored_emails = session.get('authored_emails', [])

    tfidf_score = None
    gemini_score = None
    gemini_explanation = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add':
            email = request.form.get('email')
            if email:
                authored_emails.append(email)
                session['authored_emails'] = authored_emails

        elif action == 'compare':
            new_email = request.form.get('new_email')

            tfidf_score = compare_styles_tfidf(authored_emails, new_email)
            gemini_score, gemini_explanation = compare_styles_gemini(authored_emails, new_email)

    return render_template(
        'index.html',
        emails=authored_emails,
        tfidf_score=tfidf_score,
        gemini_score=gemini_score,
        gemini_explanation=gemini_explanation
    )

    return render_template('index.html', emails=authored_emails)

@app.route('/clear')
def clear():
    session.pop('authored_emails', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)