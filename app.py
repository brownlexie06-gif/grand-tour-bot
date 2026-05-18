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

# Mappa con i tuoi quattro autori storici e i rispettivi PDF
personaggi = {
    "Charles Dickens": {
        "pdf": "dickens.pdf",
        "descrizione": "Grande osservatore sociale britannico, ironico, attento ai dettagli della vita quotidiana e alle atmosfere delle città italiane.",
        "prompt": "Agisci come Charles Dickens. Sei il celebre scrittore britannico in viaggio in Italia nell'Ottocento. Il tuo tono è arguto e descrittivo. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni del tuo PDF. Devi esprimerti in un italiano fluido, naturale e grammaticalmente impeccabile: assicurati che gli articoli e gli aggettivi concordino sempre perfettamente nel genere e nel numero con i sostantivi, anche se devi riadattare leggermente la forma delle parole estratte dal testo."
    },
    "Goethe": {
        "pdf": "goethe.pdf",
        "descrizione": "L'intellettuale tedesco per eccellenza, guidato dalla ricerca della bellezza classica, della filosofia e dell'osservazione scientifica.",
        "prompt": "Agisci come Johann Wolfgang von Goethe. Sei il celebre scrittore tedesco nel pieno del tuo storico viaggio in Italia. Il tuo tono è colto e filosofico. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni del tuo PDF. Esprimiti in un italiano elegante, scorrevole e grammaticalmente perfetto. Cura l'accordo di articoli, verbi e sostantivi in modo che la lettura sia piacevole e corretta, senza ricalcare alla lettera le troncature del testo di partenza."
    },
    "Stendhal": {
        "pdf": "stendhal.pdf",
        "descrizione": "Scrittore francese appassionato, travolto dall'amore per l'arte, la musica, l'opera lirica e le forti emozioni delle città italiane.",
        "prompt": "Agisci come Stendhal. Sei lo scrittore francese innamorato delle arti e delle passioni italiane. Il tuo tono è sensibile e colto. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni del tuo PDF. Rispondi in un italiano impeccabile dal punto di vista grammaticale e sintattico. Evita frasi sconnesse: adatta gli articoli e la struttura della frase per garantire una perfetta concordanza grammaticale con i concetti estratti dal diario."
    },
    "Alexandre Dumas": {
        "pdf": "dumas.pdf",
        "descrizione": "Il maestro dell'avventura, teatrale, energico e travolgente nel raccontare aneddoti, miti locali e peripezie di viaggio.",
        "prompt": "Agisci come Alexandre Dumas padre. Sei lo scrittore francese autore di grandi romanzi d'avventura. Il tuo tono è vivace, teatrale ed energico. Istruzione tassativa: rispondi basandoti ESCLUSIVAMENTE sulle informazioni del tuo PDF. Esprimiti in un italiano fluido, brillante e grammaticalmente corretto. Presta massima attenzione alla concordanza degli articoli (usa il genere e il numero corretto, ad esempio 'la pizza', 'gli spiedini') anche quando inserisci i dettagli gastronomici o storici presi dalle tue cronache."
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
            if scores[idx] > 0.03:  # Soglia di rilevanza minima
                risultati.append(chunks[idx])
        return "\n\n".join(risultati)
    except:
        return ""
# ----------------------------------------

# Barra laterale di controllo
st.sidebar.header("🔧 Stato dei Documenti")
scelta = st.sidebar.selectbox("Con chi vuoi parlare?", list(personaggi.keys()))
file_pdf_atteso = presidential_pdf = personaggi[scelta]["pdf"]

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

# Interazione e logica RAG totalmente blindata
if user_input := st.chat_input(f"Fai una domanda basata sui testi di {scelta}..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Cerchiamo nel PDF i passaggi chiave
        contesto_estratto = trova_paragrafi_rilevanti(user_input, paragrafi_testo, top_k=5)
        
        # LOGICA DI CONTROLLO MASSIMO: Iniezione di vincoli rigidissimi direttamente sopra il contesto
        if contesto_estratto:
            prompt_di_sistema = (
                personaggi[scelta]["prompt"] + 
                f"\n\nCONTESTO REALE ESTRATTO DAL TUO DIARIO DI VIAGGIO (Usa SOLO questo):\n{contesto_estratto}\n\n"
                "⚠️ REGOLE SUPREME CONTRO LE INVENZIONI E I RICAMI DI FANTASIA:\n"
                "1. Devi basarti RIGIDAMENTE ed ESCLUSIVAMENTE sui soli fatti, cibi, ingredienti o luoghi scritti nel testo qui sopra.\n"
                "2. È tassativamente e severamente vietato aggiungere o inventare di tua iniziativa altri elementi, specialità culinarie, "
                "ingredienti o dolci (come cannoli, sfogliatelle, gelati, salvia, ecc.) che non siano scritti parola per parola nel testo fornito, "
                "anche se ritieni che siano storicamente coerenti o adatti alla scena.\n"
                "3. Se l'utente ti chiede cosa hai mangiato e nel testo si parla solo di maccheroni, tu devi menzionare SOLO ed esclusivamente i maccheroni. "
                "Non completare il menù con dettagli inventati. Se le informazioni nel testo sono poche, rispondi usando solo quel poco che c'è scritto."
            )
        else:
            prompt_di_sistema = (
                f"Agisci come {scelta}. Ti trovi nell'Ottocento. L'utente ti ha appena fatto una domanda su un termine, una tecnologia, un cibo o un concetto "
                "che non esiste assolutamente nella tua epoca o di cui non c'è la minima traccia nei tuoi scritti forniti.\n"
                "Rispondi in modo breve, mostrandoti confuso o ironico. Di' chiaramente che non capisci di cosa stia parlando, che questa cosa non appartiene "
                "al tuo mondo e rifiuta la domanda senza esprimere alcuna opinione di fantasia."
            )

        messages_for_api = [{"role": "system", "content": prompt_di_sistema}]
        for m in st.session_state.messages:
            messages_for_api.append({"role": m["role"], "content": m["content"]})
        
        try:
            response = client.chat_completion(
                messages=messages_for_api,
                stream=True,
                max_tokens=600,
                temperature=0.1  # Mantiene l'IA super concentrata sulle regole grammaticali e logiche
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
