import streamlit as st
import os
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurazione della pagina web
st.set_page_config(page_title="Grand Tour Portal - Chatbot", page_icon="🏛️", layout="wide")

st.title("🏛️ Grand Tour Portal")
st.subheader("Digital Storytelling Project - Autori del Grand Tour")

# Lettura del token dai Secrets di Streamlit
HF_TOKEN = st.secrets.get("HF_TOKEN", "")

# Definiamo il modello di intelligenza artificiale (Qwen)
MODELLO_OPEN_SOURCE = "Qwen/Qwen2.5-7B-Instruct"

# Inizializzazione del client ufficiale di Hugging Face
client = InferenceClient(model=MODELLO_OPEN_SOURCE, token=HF_TOKEN)

# Mappa con i quattro autori storici e le rispettive lingue dei PDF
personaggi = {
    "Charles Dickens": {
        "pdf": "dickens.pdf",
        "lingua_pdf": "inglese",
        "descrizione": "Grande osservatore sociale britannico, ironico, attento ai dettagli della vita quotidiana e alle atmosfere delle città italiane.",
        "prompt": "Agisci come Charles Dickens. Sei il celebre scrittore britannico in viaggio in Italia nell'Ottocento. Il tuo tono è arguto, descrittivo, venato di sottile ironia britannica e profondamente attento ai costumi e alle scene di vita quotidiana. Rispondi in italiano con eleganza."
    },
    "Goethe": {
        "pdf": "goethe.pdf",
        "lingua_pdf": "italiano",
        "descrizione": "L'intellettuale tedesco per excellenza, guidato dalla ricerca della bellezza classica, della filosofia e dell'osservazione scientifica.",
        "prompt": "Agisci come Johann Wolfgang von Goethe. Sei il celebre scrittore e scienziato tedesco nel pieno del tuo storico viaggio in Italia. Il tuo tono è colto, filosofico, analitico e innamorato dell'arte classica e della natura mediterranea. Rispondi in italiano."
    },
    "Stendhal": {
        "pdf": "stendhal.pdf",
        "lingua_pdf": "inglese",
        "descrizione": "Scrittore francese appassionato, travolto dall'amore per l'arte, la musica, l'opera lirica e le forti emozioni delle città italiane.",
        "prompt": "Agisci come Stendhal (Marie-Henri Beyle). Sei lo scrittore francese perdutamente innamorato dell'Italia, della sua musica e dei suoi capolavori artistici. Il tuo tono è appassionato, emotivo, sensibile e colto. Rispondi in italiano."
    },
    "Alexandre Dumas": {
        "pdf": "dumas.pdf",
        "lingua_pdf": "francese",
        "descrizione": "Il maestro dell'avventura, teatrale, energico e travolgente nel raccontare aneddoti, miti locali e peripezie di viaggio.",
        "prompt": "Agisci come Alexandre Dumas padre. Sei lo scrittore francese autore di grandi romanzi d'avventura, in viaggio in Italia. Il tuo tono è vivace, teatrale, energico, ricco di spirito d'avventura e amore per le storie avvincenti. Rispondi in italiano."
    }
}

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

def trova_paragrafi_rilevanti(query, chunks, top_k=5):
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
            if scores[idx] > 0.05:  # Soglia di rilevanza matematica rigorosa
                risultati.append(chunks[idx])
        return "\n\n".join(risultati)
    except:
        return ""

# INTERPRETE SIMULTANEO: Traduce la domanda nella lingua corretta prima di interrogare il PDF
def traduci_query_per_pdf(testo_italiano, lingua_destinazione):
    if lingua_destinazione == "italiano":
        return testo_italiano
    try:
        prompt_traduzione = f"Traduci la seguente domanda in {lingua_destinazione} in modo semplice e diretto, usando i termini storici appropriati. Restituisci SOLO la traduzione senza commenti aggiuntivi: {testo_italiano}"
        res = client.chat_completion(
            messages=[{"role": "user", "content": prompt_traduzione}],
            max_tokens=60
        )
        return res.choices[0].message.content.strip()
    except:
        return testo_italiano
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

# Interazione e logica RAG multilingue blindata
if user_input := st.chat_input(f"Fai una domanda basata sui testi di {scelta}..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Individuiamo la lingua del documento dell'autore corrente
        lingua_target = personaggi[scelta]["lingua_pdf"]
        
        # Traduzione dinamica e nascosta per la ricerca nei capitoli stranieri
        stringa_da_cercare = traduci_query_per_pdf(user_input, lingua_target)
        
        # Cerchiamo nel PDF usando i termini linguistici appropriati
        contesto_estratto = trova_paragrafi_rilevanti(stringa_da_cercare, paragrafi_testo, top_k=5)
        
        # Il guardrail logico contro le invenzioni di memoria
        if contesto_estratto:
            prompt_di_sistema = (
                personaggi[scelta]["prompt"] + 
                f"\n\nCONTESTO REALE ESTRATTO DAL TUO DIARIO (Nota: il testo è in {lingua_target}, analizzalo con attenzione ma rispondi ed esprimiti unicamente in lingua italiana):\n{contesto_estratto}\n\n"
                "⚠️ REGOLE SUPREME DI COMPORTAMENTO:\n"
                "1. Rispondi basandoti RIGIDAMENTE ed ESCLUSIVAMENTE sui soli fatti, elementi o luoghi presenti nel testo sopra.\n"
                "2. È severamente vietato aggiungere dettagli di fantasia, ingredienti o aneddoti non scritti esplicitamente nel diario.\n"
                "3. Esprimiti in un italiano fluido, naturale e grammaticalmente corretto, curando la concordanza di articoli e aggettivi."
            )
        else:
            prompt_di_sistema = (
                f"Agisci come {scelta}. Ti trovi nell'Ottocento. L'utente ti ha fatto una domanda su un concetto, "
                "un'invenzione moderna o un dettaglio che non è assolutamente presente nei tuoi diari forniti.\n"
                "Rispondi in modo molto breve, mostrandoti confuso o ironico. Di' chiaramente che non ricordi questo dettaglio "
                "o che non trovi alcuna traccia di questa cosa nelle tue cronache di viaggio, rifiutando di inventare opinioni."
            )

        messages_for_api = [{"role": "system", "content": prompt_di_sistema}]
        for m in st.session_state.messages:
            messages_for_api.append({"role": m["role"], "content": m["content"]})
        
        try:
            response = client.chat_completion(
                messages=messages_for_api,
                stream=True,
                max_tokens=600,
                temperature=0.1
            )
            
            full_response = ""
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Errore tecnico nella generazione della risposta: {e}")
