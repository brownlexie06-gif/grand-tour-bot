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
        "descrizione": "L'intellettuale tedesco per eccellenza, guidato dalla ricerca della bellezza classica, della filosofia e dell'osservazione scientifica.",
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

def trova_paragrafi_rilevanti(query, chunks, top_k=3):
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
            if scores[idx] > 0.05:  # Filtro di pertinenza
                risultati.append(chunks[idx])
        return "\n\n".join(risultati)
    except:
        return ""

# TRADUTTORE DELLA SOLA QUERY UTENTE (Breve e preciso)
def traduci_domanda_utente(testo_italiano, lingua_destinazione):
    if lingua_destinazione == "italiano":
        return testo_italiano
    try:
        res = client.chat_completion(
            messages=[
                {"role": "system", "content": "Sei un traduttore automatico. Traduci la domanda dell'utente nella lingua richiesta in modo diretto. Restituisci SOLO la traduzione, senza commenti."},
                {"role": "user", "content": f"Traduci in {lingua_destinazione}: {testo_italiano}"}
            ],
            max_tokens=100,
            temperature=0.1
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

# Interazione e logica RAG ad alte prestazioni
if user_input := st.chat_input(f"Fai una domanda basata sui testi di {scelta}..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        lingua_target = personaggi[scelta]["lingua_pdf"]
        
        # Traduciamo solo la stringa di ricerca per il TF-IDF
        stringa_da_cercare = traduci_domanda_utente(user_input, lingua_target)
        
        # Estraiamo il contesto in LINGUA ORIGINALE (evita alterazioni e allucinazioni da traduzione)
        contesto_estratto = trova_paragrafi_rilevanti(stringa_da_cercare, paragrafi_testo, top_k=3)
        
        # IL GUARDRAIL LOGICO CON TRIPLE VIRGOLETTE
        if contesto_estratto:
            prompt_di_sistema = f"""{personaggi[scelta]["prompt"]}

[CONTESTO AUTENTICO IN LINGUA {lingua_target.upper()} ESTRATTO DAL TUO DIARIO]:
{contesto_estratto}

⚠️ DIRETTIVE DI VERIDICITÀ ASSOLUTA (TOLLERANZA ZERO PER LE INVENZIONI):
1. Tu comprendi perfettamente la lingua {lingua_target} del testo sopra riportato, ma devi formulare la tua risposta unicamente in un italiano fluido, naturale ed elegante.
2. La tua unica ed esclusiva fonte di verità sono i fatti scritti nel testo in lingua originale sopra riportato.
3. ISOLAMENTO DELLA CONOSCENZA: Se l'utente ti nomina o ti interroga su piatti, ingredienti, luoghi o dettagli (come pasta alla puttanesca, cocchi, caponotti, arrosti o altro) che NON sono esplicitamente scritti nel testo originale sopra, devi dichiarare che nei tuoi diari non c'è traccia di queste cose. Non usare la tua immaginazione per confermare falsi miti.
4. Riporta solo i fatti presenti, senza estrapolare, senza inventare ricette e senza aggiungere aggettivi qualificativi di testa tua."""
        else:
            prompt_di_sistema = f"""Agisci come {scelta}. Ti trovi nell'Ottocento.

Istruzione universale di vuoto documentale: L'utente ti ha fatto una domanda su un concetto, un'invenzione moderna, un cibo o un dettaglio che non è assolutamente presente nei tuoi scritti forniti o che non appartiene al tuo secolo.
Rispondi in modo molto breve (massimo due frasi), dichiarando con fermezza storica o ironia che non ricordi questo elemento, che non è registrato nelle tue cronache o che non fa parte del tuo mondo. Rifiuta la domanda senza inventare nulla."""

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
