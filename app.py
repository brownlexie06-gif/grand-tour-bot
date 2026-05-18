import streamlit as st
from openai import OpenAI
import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurazione della pagina web
st.set_page_config(page_title="Grand Tour Portal - Chatbot", page_icon="🏛️", layout="wide")

st.title("🏛️ Grand Tour Portal")
st.subheader("Digital Storytelling Project - Autori del Grand Tour")

# Lettura del token dai Secrets di Streamlit
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# 1. Definiamo prima il modello di intelligenza artificiale
MODELLO_OPEN_SOURCE = "Qwen/Qwen2.5-7B-Instruct"

# 2. Inseriamo il modello direttamente nel link web (base_url) come richiesto ora da Hugging Face
client = OpenAI(
    base_url=f"https://api-inference.huggingface.co/models/{MODELLO_OPEN_SOURCE}/v1",
    api_key=HF_TOKEN
)

# Mappa con i tuoi quattro autori storici
personaggi = {
    "Charles Dickens": {
        "pdf": "dickens.pdf",
        "descrizione": "Grande osservatore sociale britannico, ironico, attento ai dettagli della vita quotidiana e alle atmosfere delle città italiane.",
        "prompt": "Agisci come Charles Dickens. Sei il celebre scrittore britannico in viaggio in... [truncated for spacing, use full prompts below]"
    },
    "Goethe": {
        "pdf": "goethe.pdf",
        "descrizione": "L'intellettuale tedesco per eccellenza, guidato dalla ricerca della bellezza classica, della filosofia e dell'osservazione scientifica.",
        "prompt": "Agisci come Johann Wolfgang von Goethe. Sei il celebre scrittore e scienziato tedesco nel pieno del suo storico viaggio in Italia..."
    },
    "Stendhal": {
        "pdf": "stendhal.pdf",
        "descrizione": "Scrittore francese appassionato, travolto dall'amore per l'arte, la musica, l'opera lirica e le forti emozioni delle città italiane.",
        "prompt": "Agisci come Stendhal (Marie-Henri Beyle). Sei lo scrittore francese perdutamente innamorato dell'Italia..."
    },
    "Alexandre Dumas": {
        "pdf": "dumas.pdf",
        "descrizione": "Il maestro dell'avventura, teatrale, energico e travolgente nel raccontare aneddoti, miti locali e peripezie di viaggio.",
        "prompt": "Agisci come Alexandre Dumas padre. Sei lo scrittore francese autore di grandi romanzi d'avventura..."
    }
}

# Ripristino dei testi completi dei prompt per sicurezza
personaggi["Charles Dickens"]["prompt"] = "Agisci come Charles Dickens. Sei il celebre scrittore britannico in viaggio in Italia. Il tuo tono è arguto, descrittivo, venato di sottile ironia britannica e profondamente attento ai costumi e alle scene di vita quotidiana. Rispondi in italiano con eleganza. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni presenti nel CONTESTO fornito. Se la risposta non è presente nel contesto, di' chiaramente che non trovi questo aneddoto nei tuoi diari italiani. Non inventare nulla."
personaggi["Goethe"]["prompt"] = "Agisci come Johann Wolfgang von Goethe. Sei il celebre scrittore e scienziato tedesco nel pieno del suo storico viaggio in Italia. Il tuo tono è colto, filosofico, analitico e innamorato dell'arte classica e della natura mediterranea. Rispondi in italiano. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni presenti nel CONTESTO fornito. Se la risposta non è presente nel contesto, ammetti chiaramente che non fa parte delle tue osservazioni documentate."
personaggi["Stendhal"]["prompt"] = "Agisci come Stendhal (Marie-Henri Beyle). Sei lo scrittore francese perdutamente innamorato dell'Italia, della sua musica e dei suoi capolavori artistici. Il tuo tono è appassionato, emotivo, sensibile e colto. Rispondi in italiano. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni presenti nel CONTESTO fornito. Se la risposta non è presente nel contesto, di' che la prima anima non ha annotato questo dettaglio nei diari."
personaggi["Alexandre Dumas"]["prompt"] = "Agisci come Alexandre Dumas padre. Sei lo scrittore francese autore di grandi romanzi d'avventura, in viaggio in Italia. Il tuo tono è vivace, teatrale, energico, ricco di spirito d'avventura e amore per le storie avvincenti. Rispondi in italiano. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni presenti nel CONTESTO fornito. Se la risposta non è presente nel contesto, di' che questa storia non fa parte delle tue cronache di viaggio."

# --- FUNZIONI PER GESTIRE I PDF (RAG) ---
@st.cache_data
def carica_e_spezzetta_pdf(nome_file, chunk_size=600):
    if not os.path.exists(nome_file):
        return []
    try:
        reader = PdfReader(nome_file)
        testo_completo = ""
        for pagina in reader.pages:
            testo_estratto = pagina.extract_text()
            if testo_estratto:
                testo_completo += testo_estratto + "\n"
        
        parole = testo_completo.split()
        chunks = []
        for i in range(0, len(parole), chunk_size - 100):
            chunk = " ".join(parole[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    except:
        return []

def trova_paragrafi_rilevanti(query, chunks, top_k=2):
    if not chunks:
        return ""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        
        risultati = []
        for idx in top_indices:
            if scores[idx] > 0.02:
                risultati.append(chunks[idx])
        return "\n\n".join(risultati)
    except:
        return ""
# ----------------------------------------

# Barra laterale di controllo
st.sidebar.header("🔧 Stato dei Documenti")
scelta = st.sidebar.selectbox("Con chi vuoi parlare?", list(personaggi.keys()))
file_pdf_atteso = personaggi[scelta]["pdf"]

if os.path.exists(file_pdf_atteso):
    st.sidebar.success(f"📚 Fonte '{file_pdf_atteso}' rilevata con successo!")
    paragrafi_testo = carica_e_spezzetta_pdf(file_pdf_atteso)
    st.sidebar.info(f"Testo suddiviso in {len(paragrafi_testo)} sezioni consultabili.")
else:
    st.sidebar.error(f"❌ File '{file_pdf_atteso}' non trovato su GitHub. L'autore risponderà senza base documentale.")
    paragrafi_testo = []

st.sidebar.write("---")
if st.sidebar.button("Cancella Cronologia Chat"):
    st.session_state.messages = []

# Gestione memoria chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ultimo_personaggio" not in st.session_state:
    st.session_state.ultimo_personaggio = scelta
elif st.session_state.ultimo_personaggio != scelta:
    st.session_state.messages = []
    st.session_state.ultimo_personaggio = scelta

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interazione e logica RAG
if user_input := st.chat_input(f"Fai una domanda basata sui testi di {scelta}..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        contesto_estratto = trova_paragrafi_rilevanti(user_input, paragrafi_testo, top_k=2)
        
        prompt_di_sistema = personaggi[scelta]["prompt"]
        if contesto_estratto:
            prompt_di_sistema += f"\n\nCONTESTO ESTRATTO DAL TUO PDF:\n{contesto_estratto}"
        else:
            prompt_di_sistema += "\n\nATTENZIONE: Nessun dato rilevante trovato nel tuo PDF per questa specifica domanda."

        messages_for_api = [{"role": "system", "content": prompt_di_sistema}]
        for m in st.session_state.messages:
            messages_for_api.append({"role": m["role"], "content": m["content"]})
        
        try:
            response = client.chat.completions.create(
                model=MODELLO_OPEN_SOURCE,
                messages=messages_for_api,
                stream=True,
                max_tokens=400
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Errore tecnico nella generazione della risposta: {e}")
