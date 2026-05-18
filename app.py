import streamlit as st
import os
import numpy as np
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Grand Tour Portal - Chatbot", page_icon="🏛️", layout="wide")

st.title("🏛️ Grand Tour Portal")
st.subheader("Digital Storytelling Project - Autori del Grand Tour")

HF_TOKEN = st.secrets.get("HF_TOKEN", "")
MODELLO_OPEN_SOURCE = "Qwen/Qwen2.5-7B-Instruct"

client = InferenceClient(model=MODELLO_OPEN_SOURCE, token=HF_TOKEN)

# ---------------- PERSONAGGI ----------------
personaggi = {
    "Charles Dickens": {
        "pdf": "dickens.pdf",
        "descrizione": "Osservatore sociale britannico, ironico e descrittivo.",
        "prompt": "Agisci come Charles Dickens. Sei in viaggio in Italia. Il tuo tono è arguto, descrittivo e ironico. Rispondi in italiano."
    },
    "Goethe": {
        "pdf": "goethe.pdf",
        "descrizione": "Intellettuale tedesco, filosofico e analitico.",
        "prompt": "Agisci come Johann Wolfgang von Goethe. Tono colto, filosofico, analitico. Rispondi in italiano."
    },
    "Stendhal": {
        "pdf": "stendhal.pdf",
        "descrizione": "Scrittore emotivo e appassionato.",
        "prompt": "Agisci come Stendhal. Tono appassionato, emotivo e sensibile. Rispondi in italiano."
    },
    "Alexandre Dumas": {
        "pdf": "dumas.pdf",
        "descrizione": "Narratore teatrale e avventuroso.",
        "prompt": "Agisci come Alexandre Dumas padre. Tono vivace, teatrale e avventuroso. Rispondi in italiano."
    }
}

# ---------------- PDF + CHUNKING SICURO A PAROLE ----------------
@st.cache_data
def carica_e_spezzetta_pdf(nome_file, chunk_size=200, overlap=50):
    if not os.path.exists(nome_file):
        return []

    try:
        reader = PdfReader(nome_file)
        testo_completo = ""

        for pagina in reader.pages:
            t = pagina.extract_text()
            if t:
                testo_completo += t + " "

        # Sminuzzamento a parole per evitare limiti di token
        parole = testo_completo.split()
        chunks = []
        for i in range(0, len(parole), chunk_size - overlap):
            chunk = " ".join(parole[i:i+chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk)

        return chunks

    except Exception as e:
        st.error(f"Errore PDF: {e}")
        return []

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def build_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return model, embeddings

# ---------------- RETRIEVAL ----------------
def trova_paragrafi_rilevanti(query, chunks, model, embeddings, top_k=3):
    if not chunks or model is None or embeddings is None:
        return ""

    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(scores)[-top_k:][::-1]

    risultati = [chunks[i] for i in top_indices if scores[i] > 0.35]

    return "\n\n".join(risultati)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔧 Stato dei Documenti")
scelta = st.sidebar.selectbox("Con chi vuoi parlare?", list(personaggi.keys()))
file_pdf_atteso = personaggi[scelta]["pdf"]

paragrafi_testo = carica_e_spezzetta_pdf(file_pdf_atteso)

if paragrafi_testo:
    model_embed, embeddings = build_index(paragrafi_testo)
    st.sidebar.success(f"📚 {len(paragrafi_testo)} sezioni indicizzate")
else:
    model_embed, embeddings = None, None
    st.sidebar.error(f"❌ File '{file_pdf_atteso}' non trovato")

st.sidebar.write("---")

if st.sidebar.button("Cancella Cronologia Chat"):
    st.session_state.messages = []

# ---------------- MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ultimo_personaggio" not in st.session_state:
    st.session_state.ultimo_personaggio = scelta
elif st.session_state.ultimo_personaggio != scelta:
    st.session_state.messages = []
    st.session_state.ultimo_personaggio = scelta

# Mostra chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CHAT ----------------
if user_input := st.chat_input(f"Fai una domanda a {scelta}..."):

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # 🔍 Retrieval semantico
        contesto_estratto = trova_paragrafi_rilevanti(
            user_input,
            paragrafi_testo,
            model_embed,
            embeddings
        )

        # 🧠 Prompt blindato contro le invenzioni
        if contesto_estratto:
            prompt_di_sistema = f"""{personaggi[scelta]["prompt"]}

[CONTESTO AUTENTICO ESTRATTO DAL TUO DIARIO]:
{contesto_estratto}

⚠️ DIRETTIVE DI VERIDICITÀ ASSOLUTA:
1. La tua unica ed esclusiva fonte di verità sono i fatti scritti nel testo originale qui sopra.
2. ISOLAMENTO: Ignora qualsiasi conoscenza globale. Non inventare, non presumere.
3. Se l'utente ti chiede dettagli che NON sono esplicitamente scritti nel testo sopra, dichiara che non ne hai traccia nei tuoi diari.
4. Rispondi in italiano in modo naturale."""
        else:
            prompt_di_sistema = f"""Agisci come {scelta}. Ti trovi nell'Ottocento.

Rispondi in massimo due frasi dichiarando con fermezza che non ricordi questo dettaglio o che non fa parte delle tue cronache di viaggio. Rifiuta la domanda senza inventare nulla."""

        # Costruisce i messaggi
        messages_for_api = [{"role": "system", "content": prompt_di_sistema}]
        for m in st.session_state.messages[-6:]:
            messages_for_api.append(m)

        try:
            response = client.chat_completion(
                messages=messages_for_api,
                stream=True,
                max_tokens=500,
                temperature=0.1  # <-- Fondamentale per bloccare la fantasia
            )

            full_response = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(f"Errore tecnico: {e}")
