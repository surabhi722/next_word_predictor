import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page config (must be first Streamlit call)
# ------------------------------
st.set_page_config(page_title="Next Word Oracle", page_icon="✨", layout="centered")

# ------------------------------
# Custom CSS for a colorful, quote-themed look
# ------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #831843 100%);
    }
    h1, h2, h3, p, label, .stMarkdown, .stCaption {
        color: #f5f3ff !important;
    }
    .quote-card {
        background: rgba(255, 255, 255, 0.08);
        border-left: 5px solid #f472b6;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 20px 0;
        backdrop-filter: blur(6px);
    }
    .quote-text {
        font-family: Georgia, 'Times New Roman', serif;
        font-style: italic;
        font-size: 22px;
        color: #fdf4ff;
        line-height: 1.5;
    }
    .predicted-word {
        color: #fbbf24;
        font-weight: 700;
    }
    .stTextInput input {
        background-color: rgba(255,255,255,0.12) !important;
        color: #fff !important;
        border: 1px solid #a78bfa !important;
        border-radius: 8px !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #f472b6, #a78bfa);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #ec4899, #8b5cf6);
        color: white;
    }
    section[data-testid="stSidebar"] {
        background: rgba(0,0,0,0.25);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_word

model, tokenizer, max_len, index_word = load_resources()

# ------------------------------
# Prediction helpers
# ------------------------------
def get_prediction_probs(text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')
    preds = model.predict(sequence, verbose=0)[0]
    return preds

def top_k_words(preds, k=3):
    top_indices = np.argsort(preds)[-k:][::-1]
    return [(index_word.get(i, ""), float(preds[i])) for i in top_indices if i in index_word]

def generate_words(text, n_words):
    current_text = text
    generated = []
    for _ in range(n_words):
        preds = get_prediction_probs(current_text)
        next_index = np.argmax(preds)
        next_word = index_word.get(next_index, "")
        if not next_word:
            break
        generated.append(next_word)
        current_text += " " + next_word
    return generated

# ------------------------------
# Sidebar settings
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    n_words = st.slider("Words to generate", 1, 20, 5)
    show_probs = st.checkbox("Show top-3 word probabilities", value=True)
    st.markdown("---")
    st.caption("Model: LSTM trained on 3,000 quotes")

# ------------------------------
# Main UI
# ------------------------------
st.title("✨ Next Word Oracle")
st.write("Type a phrase and let the LSTM continue it, quote-style.")

user_input = st.text_input("✍️ Start your quote:", placeholder="The only way to do great work is...")

col1, col2 = st.columns(2)
with col1:
    predict_clicked = st.button("🔮 Predict Next Word")
with col2:
    generate_clicked = st.button("🪄 Generate Full Quote")

if predict_clicked or generate_clicked:
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        if predict_clicked:
            preds = get_prediction_probs(user_input)
            next_word = top_k_words(preds, 1)[0][0]

            st.markdown(f"""
            <div class="quote-card">
                <div class="quote-text">"{user_input} <span class="predicted-word">{next_word}</span>"</div>
            </div>
            """, unsafe_allow_html=True)

            if show_probs:
                st.subheader("Top 3 candidates")
                top3 = top_k_words(preds, 3)
                df = pd.DataFrame(top3, columns=["word", "probability"]).set_index("word")
                st.bar_chart(df)

        if generate_clicked:
            generated = generate_words(user_input, n_words)
            full_quote = f"{user_input} " + " ".join(generated)

            st.markdown(f"""
            <div class="quote-card">
                <div class="quote-text">"{full_quote}"</div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction • Trained on 3,000 quotes • Built with Streamlit")