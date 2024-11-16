import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
# Download necessary NLTK data for tokenization
nltk.download('punkt')
def extract_keywords(text, num_keywords=8):
    """Extracts a diverse set of keywords from the text based on term frequency."""
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    words = vectorizer.get_feature_names_out()
    # Get the frequency of each word
    freq = X.toarray().sum(axis=0)
    # Pair words with their frequencies and sort
    keywords = sorted(zip(words, freq), key=lambda x: x[1], reverse=True)
    # Return the top keywords
    return [kw[0] for kw in keywords[:num_keywords]]
def ai_notes_generator(input_text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(input_text)
    print("Tokenized Sentences:", sentences)
    # Check if input text is long enough for summarization
    if len(sentences) < 2:
        return "Text too short for summarization. Please provide more content."
    # Load the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Generate a summary of the input text
    summary = summarizer(input_text, max_length=200, min_length=50, do_sample=False)
    # Extract the summary text
    summary_text = summary[0]['summary_text']
    
    # Extract dynamic keywords from the input text
    keywords = extract_keywords(input_text)
    # Integrate the keywords into the summary
    if keywords:
        keyword_section = "Key aspects to consider include: " + ", ".join(keywords) + "."
        summary_text = f"{summary_text} Furthermore, {keyword_section}"
    return summary_text
# Input text (can handle any user input related to various topics)
input_text = """
An engine is a machine designed to convert energy into mechanical force or motion. It typically works by harnessing a specific energy source—such as fuel (in the case of internal combustion engines) or electricity (in electric motors)—to generate power. This power is then used to drive machinery or vehicles, enabling them to perform work. Engines are central components in a wide range of systems, from cars and airplanes to industrial machinery and power plants. Depending on the type, engines may use different principles, like combustion, steam, or electromagnetic forces, to produce motion or perform tasks.



."""
# Generate notes
notes = ai_notes_generator(input_text)
print("Generated Notes:\n", notes)