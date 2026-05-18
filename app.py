import streamlit as st
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
 
# ---------------- CONFIG ----------------
st.set_page_config(page_title="Grand Tour Portal - Chatbot", page_icon="🏛️", layout="wide")
 
st.title("🏛️ Grand Tour Portal")
st.subheader("Digital Storytelling Project - Autori del Grand Tour")
 
# API Key gestita implicitamente tramite st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(
    model="gpt-4o", 
    base_url="https://api-gpt.jrc.ec.europa.eu/v1", 
    temperature=0.6
)
 
# ---------------- PERSONAGGI ----------------
personaggi = {
    "Charles Dickens": {
        "pdf": "dickens.pdf",
        "descrizione": "Osservatore sociale britannico, ironico e descrittivo.",
        "prompt": """Agisci come Charles Dickens. Sei in viaggio in Italia.
Il tuo tono è arguto, descrittivo e ironico. Rispondi in italiano."""
    },
    "Goethe": {
        "pdf": "goethe.pdf",
        "descrizione": "Intellettuale tedesco, filosofico e analitico.",
        "prompt": """Agisci come Johann Wolfgang von Goethe.
Tono colto, filosofico, analitico. Rispondi in italiano."""
    },
    "Stendhal": {
        "pdf": "stendhal.pdf",
        "descrizione": "Scrittore emotivo e appassionato.",
        "prompt": """Agisci come Stendhal.
Tono appassionato, emotivo e sensibile. Rispondi in italiano."""
    },
    "Alexandre Dumas": {
        "pdf": "dumas.pdf",
        "descrizione": "Narratore teatrale e avventuroso.",
        "prompt": """Agisci come Alexandre Dumas padre.
Tono vivace, teatrale e avventuroso. Rispondi in italiano."""
    }
}
 
# ---------------- PDF + FIXED CHUNKING ----------------
@st.cache_data
def carica_e_spezzetta_pdf(nome_file):
    if not os.path.exists(nome_file):
        return []
 
    try:
        reader = PdfReader(nome_file)
        testo = ""
 
        for pagina in reader.pages:
            t = pagina.extract_text()
            if t:
                testo += t + "\n"
 
        # Suddivisione fissa a finestre di caratteri
        chunk_size = 1500  # ~350 token per chunk
        overlap = 200
        chunks = []
        for i in range(0, len(testo), chunk_size - overlap):
            chunk = testo[i:i + chunk_size].strip()
            if len(chunk) > 100:
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
    if not chunks or model is None:
        return ""
 
    # Genera embedding della query come vettore 1D
    query_embedding = model.encode(query, convert_to_numpy=True)
 
    # Calcolo corretto della similarità coseno (vettorizzato)
    scores = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
 
    top_indices = np.argsort(scores)[-top_k:][::-1]
 
    # Soglia flessibile a 0.30 per catturare il contesto pertinente
    risultati = [chunks[i] for i in top_indices if scores[i] > 0.30]
 
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
 
# Mostra la chat a schermo
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
 
        # Estrazione del contesto
        contesto_estratto = trova_paragrafi_rilevanti(
            user_input,
            paragrafi_testo,
            model_embed,
            embeddings
        )
 
        # Costruzione del prompt di sistema
        prompt_di_sistema = personaggi[scelta]["prompt"] + """
 
ISTRUZIONI IMPORTANTI:
- Usa le informazioni del contesto per rispondere alla domanda.
- Se il contesto non contiene l'informazione, usa le tue conoscenze storiche per rispondere rimanendo nel personaggio.
- Mantieni sempre in modo rigoroso lo stile, la personalità e il tono del personaggio storico.
"""
 
        if contesto_estratto:
            prompt_di_sistema += f"\n\nCONTESTO DI RIFERIMENTO:\n---------------------\n{contesto_estratto}\n---------------------\n"
 
        # CORREZIONE CHIAVE: Traduzione dei ruoli per lo schema LangChain ("assistant" -> "ai")
        messages_for_api = [("system", prompt_di_sistema)]
        for m in st.session_state.messages[-6:]:
            ruolo_langchain = "ai" if m["role"] == "assistant" else "user"
            messages_for_api.append((ruolo_langchain, m["content"]))
 
        try:
            full_response = ""
            # Esecuzione dello streaming con i ruoli mappati correttamente
            for chunk in llm.stream(messages_for_api):
                if chunk.content:
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")
 
            message_placeholder.markdown(full_response)
 
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
 
        except Exception as e:
            st.error(f"Errore durante la generazione: {e}")

 
