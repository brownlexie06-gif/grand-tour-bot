import streamlit as st
from openai import OpenAI

# Configurazione della pagina web
st.set_page_config(page_title="Grand Tour Portal - Chatbot", page_icon="🏛️", layout="wide")

st.title("🏛️ Grand Tour Portal")
st.subheader("Digital Storytelling Project - Modello Open Source Gratuito")

# Configuriamo il client per puntare ai server open-source gratuiti di Hugging Face
# Il codice cercherà il token all'interno dei 'Secrets' sicuri della piattaforma di hosting
HF_TOKEN = st.secrets.get("HF_TOKEN", "INSERISCI_QUI_IL_TUO_TOKEN_HF_SE_PROVI_IN_LOCALE")

client = OpenAI(
    base_url="https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN
)

# Utilizziamo un eccellente modello open-source leggero e performante
MODELLO_OPEN_SOURCE = "meta-llama/Llama-3.2-3B-Instruct"

# Definizione dei profili dei quattro protagonisti
personaggi = {
    "Il Poeta Romantico (Lord Byron)": {
        "descrizione": "Un animo tormentato e appassionato, affascinato dalle rovine e dalla natura selvaggia.",
        "prompt": "Agisci come un raffinato Poeta Romantico inglese del diciannovesimo secolo in viaggio in Italia. Il tuo tono è malinconico, poetico, colto e profondamente emotivo. Rispondi in italiano con un registro elegante e d'altri tempi."
    },
    "L'Intellettuale Illuminato (Goethe)": {
        "descrizione": "Un uomo di scienza e lettere, focalizzato sullo studio dell'arte classica e dell'archeologia.",
        "prompt": "Agisci come un celebre scrittore e scienziato tedesco dell'Ottocento durante il suo viaggio in Italia. Sei guidato dalla ragione e dall'amore per l'arte classica. Rispondi in italiano in modo chiaro ed erudito."
    },
    "Il Giovane Aristocratico": {
        "descrizione": "Un rampollo europeo nel pieno del suo viaggio di formazione, curioso di scoprire i costumi locali.",
        "prompt": "Agisci come un giovane aristocratico europeo che sta compiendo il suo Grand Tour in Italia. Sei affascinato dalla vita mondana e dalle tradizioni. Il tuo tono è vivace, curioso ed educato. Rispondi in italiano."
    },
    "Il Pittore Paesaggista": {
        "descrizione": "Un artista visivo che cattura la luce d'Italia e le atmosfere delle vedute mediterranee.",
        "prompt": "Agisci come un pittore vedutista dell'Ottocento innamorato della luce italiana. Descrivi il mondo attraverso i colori, le sfumature e le composizioni visive. Rispondi in italiano usando un linguaggio visivo."
    }
}

# Barra laterale
st.sidebar.header("Seleziona un Viaggiatore")
scelta = st.sidebar.selectbox("Con chi vuoi parlare?", list(personaggi.keys()))
st.sidebar.markdown(f"**Profilo:** {personaggi[scelta]['descrizione']}")
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

# Input utente e generazione risposta
if user_input := st.chat_input(f"Invia un messaggio a {scelta}..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        messages_for_api = [{"role": "system", "content": personaggi[scelta]["prompt"]}]
        for m in st.session_state.messages:
            messages_for_api.append({"role": m["role"], "content": m["content"]})
        
        try:
            response = client.chat.completions.create(
                model=MODELLO_OPEN_SOURCE,
                messages=messages_for_api,
                stream=True,
                max_tokens=300
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error("Configura il token Hugging Face nei Secrets o attendi che il modello gratuito torni disponibile.")
