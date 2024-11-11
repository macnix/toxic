import streamlit as st
import streamlit.components.v1 as components
import torch
from easynmt import EasyNMT
import fasttext
from transformers import BertForSequenceClassification, BertTokenizer
# Google Analytics tracking code
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
           
tracking_code = """
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-NK1R7C1JBW"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-NK1R7C1JBW');
</script>
"""

# Insert Google Analytics tracking code into the app
components.html(tracking_code, height=0, width=0)

translation_model = EasyNMT('m2m_100_418M')
dmodel = fasttext.load_model('lid.176.bin')
# Load the saved model and tokenizer
model = BertForSequenceClassification.from_pretrained('./toxic_job_description_model')
tokenizer = BertTokenizer.from_pretrained('./toxic_job_description_tokenizer')

# Translation function
def translate_to_english(text: str, source_lang: str) -> str:
    return translation_model.translate(text, source_lang=source_lang, target_lang="en")

# Function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities[0][predicted_class].item()

# Streamlit app layout
st.title("Toxic Job Description Classification")
st.write("Enter a job description to check if it's toxic or not.")
st.write("If the text is not in English, it will take some time to analyze.")

# Text input for job description
job_description = st.text_area("Job Description:")

if st.button("Predict"):
    if job_description:
        # Step 1: Detect language
        text = str(job_description)
        text = text.replace("\n", " ")
        
        label, confidence = dmodel.predict([text], k=1)
        iso_language_code = label[0][0].replace('__label__', '')
        #source_lang = iso_language_code if iso_language_code in ['uk', 'ru'] else 'uk'

        # Step 2: If language is not English, translate
        if iso_language_code != 'en':
            translated_text = translate_to_english(text, iso_language_code )
        else:
            translated_text = text  # No translation needed if already in English

        # Step 3: Predict toxicity on the English text
        # Make prediction
        predicted_class, confidence = predict(translated_text)
        label = "Toxic" if predicted_class == 1 else "Non-Toxic"
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")
    else:
        st.error("Please enter a job description.")

# Add contact email at the bottom of the app
st.markdown("---")  # Horizontal line for separation
st.markdown("For inquiries, please contact us at: [lets.skills@hotmail.com](mailto:lets.skills@hotmail.com)")
