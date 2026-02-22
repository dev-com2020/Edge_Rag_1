# app.py
import streamlit as st
import json
import chromadb
from openai import OpenAI

# --- KONFIGURACJA --- #

# Klient Ollama do generowania odpowiedzi i embeddingów
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key='lmstudio',
)

LLM_MODEL = "qwen/qwen3-4b-thinking-2507"
EMBEDDING_MODEL = "nomic-embed-text"
MEMORY_FILE = "memory.json"

# Połącz się z istniejącą bazą ChromaDB
db = chromadb.PersistentClient(path="./vector_store")
collection = db.get_collection(name="rag_wiedza")


# --- FUNKCJE POMOCNICZE --- #

def load_memory():
    try:
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_memory(memory):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(memory, f, indent=2)


def get_response_from_llm(query, context, memory):
    system_prompt = f"""Jesteś pomocnym asystentem AI. 
    Odpowiadaj na pytania użytkownika, 
    korzystając z poniższego kontekstu i faktów o użytkowniku.

    Kontekst z bazy wiedzy:
    {context}

    Fakty o użytkowniku:
    {memory}
    """

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        stream=True
    )
    return response


# --- INTERFEJS STREAMLIT --- #

st.title("Lokalny Asystent AI")
st.caption("Ollama + RAG + Pamięć")

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state.messages = []

# Wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wejście od użytkownika
if prompt := st.chat_input("O co chcesz zapytać?"):
    # Dodaj wiadomość do historii
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- LOGIKA RAG + PAMIĘĆ --- #
    with st.chat_message("assistant"):
        # 1. Wyszukaj w bazie RAG
        response = client.embeddings.create(input=prompt, model=EMBEDDING_MODEL)
        query_embedding = response.data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2
        )
        rag_context = "\n".join(results['documents'][0])

        # 2. Załaduj pamięć o użytkowniku
        memory = load_memory()

        # 3. Wygeneruj odpowiedź (streaming)
        response_stream = get_response_from_llm(prompt, rag_context, json.dumps(memory))

        # Wyświetl odpowiedź strumieniowo
        full_response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 4. Aktualizuj pamięć (prosta ekstrakcja faktów)
        fact_extraction_prompt = f"""Przeanalizuj ostatnią wymianę:

        User: {prompt}
        Assistant: {full_response}

        Wyodrębnij nowe informacje o użytkowniku w formacie:
        klucz: wartość
        klucz: wartość

        Jeśli nic nowego → napisz dokładnie: BRAK

        Bez żadnego dodatkowego tekstu.
        """

        fact_response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": fact_extraction_prompt}],
            temperature=0.1,
            max_tokens=150,
        )

        text = fact_response.choices[0].message.content.strip()

        new_facts = {}
        if text.upper() != "BRAK":
            for line in text.splitlines():
                line = line.strip()
                if ':' not in line:
                    continue
                k, v = line.split(':', 1)
                k = k.strip().lower().replace(' ', '_')
                v = v.strip()
                if v:
                    new_facts[k] = v

        if new_facts:
            memory.update(new_facts)
            save_memory(memory)
            st.sidebar.success("Zapamiętałem nowe fakty!")
        else:
            st.sidebar.info("Brak nowych faktów.")
# Wyświetlanie aktualnej pamięci w panelu bocznym
st.sidebar.title("Pamięć o Użytkowniku")
st.sidebar.json(load_memory())