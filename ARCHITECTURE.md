# Dokumentacja Techniczna — Edge_Rag_1

## Przegląd architektury

Edge_Rag_1 to aplikacja jednowarstwowa (single-tier) działająca lokalnie. Wszystkie komponenty uruchamiane są na jednej maszynie, bez zależności od zewnętrznych serwisów chmurowych.

### Diagram komponentów

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HOST MACHINE                               │
│                                                                     │
│  ┌──────────────────────┐     ┌─────────────────────────────────┐   │
│  │   Streamlit (app.py) │     │         LM Studio               │   │
│  │                      │     │   ┌─────────────────────────┐   │   │
│  │  - Chat UI           │────▶│   │  qwen3-4b-thinking-2507  │   │   │
│  │  - RAG pipeline      │     │   │  (LLM)                  │   │   │
│  │  - Memory manager    │     │   └─────────────────────────┘   │   │
│  │                      │     │   ┌─────────────────────────┐   │   │
│  └──────────┬───────────┘     │   │  nomic-embed-text        │   │   │
│             │                 │   │  (Embedding model)       │   │   │
│             │                 │   └─────────────────────────┘   │   │
│             │                 │   REST API: localhost:1234/v1    │   │
│             │                 └─────────────────────────────────┘   │
│             │                                                        │
│  ┌──────────┴───────────┐     ┌─────────────────────────────────┐   │
│  │   ChromaDB           │     │         memory.json             │   │
│  │   (vector_store/)    │     │                                 │   │
│  │  - Persistent store  │     │  - Key-value user facts         │   │
│  │  - Collection:       │     │  - Updated after each turn      │   │
│  │    rag_wiedza        │     │                                 │   │
│  └──────────────────────┘     └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Opis modułów

### `app.py` — Główna aplikacja

| Sekcja | Linie | Opis |
|--------|-------|------|
| Konfiguracja | 1–21 | Inicjalizacja klienta OpenAI, nazwy modeli, połączenie z ChromaDB |
| `load_memory()` | 26–31 | Wczytuje `memory.json`; zwraca `{}` przy błędzie |
| `save_memory()` | 34–36 | Serializuje słownik faktów do JSON z wcięciami |
| `get_response_from_llm()` | 39–59 | Buduje system prompt, wywołuje LLM w trybie streaming |
| Interfejs Streamlit | 62–149 | Obsługa czatu, pipeline RAG+Memory, sidebar |

#### Szczegółowy przepływ w pętli czatu (`app.py:77–149`)

```
prompt (input)
    │
    ├─ 1. embeddings.create(prompt) ──────────────► query_embedding
    │
    ├─ 2. collection.query(query_embedding, n=2) ─► rag_context (2 chunks)
    │
    ├─ 3. load_memory() ──────────────────────────► memory dict
    │
    ├─ 4. get_response_from_llm(prompt, rag_context, memory)
    │         └─ stream=True ────────────────────► full_response (streaming)
    │
    └─ 5. Ekstrakcja faktów
              └─ LLM call (temperature=0.1, max_tokens=150)
              └─ parse "klucz: wartość" lines
              └─ memory.update(new_facts)
              └─ save_memory()
```

### `ingest.py` — Indeksowanie bazy wiedzy

| Sekcja | Opis |
|--------|------|
| Inicjalizacja klienta | OpenAI-compatible client → LM Studio |
| ChromaDB setup | `PersistentClient` + `get_or_create_collection("rag_wiedza")` |
| Chunking | Split po `\n`, filtracja pustych linii |
| Pętla indeksowania | Dla każdego chunka: `embeddings.create()` → `collection.add()` |

**Uwaga:** `ingest.py` używa modelu `s3dev-ai/text-embedding-nomic-embed-text-v1.5`, podczas gdy `app.py` używa `nomic-embed-text`. Należy upewnić się, że oba skrypty korzystają z tego samego modelu załadowanego w LM Studio.

---

## Modele danych

### Kolekcja ChromaDB (`rag_wiedza`)

Każdy dokument w kolekcji zawiera:

```
{
  "id": "0",                    # string, numer chunka
  "embedding": [0.1, -0.3, ...], # wektor float (rozmiar zależy od modelu)
  "document": "tekst chunka"    # oryginalny tekst
}
```

### Plik `memory.json`

Płaski słownik JSON z faktami o użytkowniku:

```json
{
  "klucz_faktu": "wartość_faktu",
  "imie": "Jan",
  "stanowisko": "developer"
}
```

Klucze są normalizowane: małe litery, spacje zamieniane na `_`.

### Historia czatu (Streamlit session_state)

Przechowywana in-memory, resetowana przy odświeżeniu strony:

```python
st.session_state.messages = [
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "..."},
]
```

---

## Integracja z LM Studio

Aplikacja korzysta z klienta `openai` Python skierowanego na lokalny endpoint LM Studio.

### Endpointy

| Endpoint | Metoda | Zastosowanie |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Generowanie odpowiedzi LLM |
| `/v1/embeddings` | POST | Generowanie wektorów semantycznych |

### Parametry wywołania LLM

**Generowanie odpowiedzi** (`get_response_from_llm`):
- `model`: `qwen/qwen3-4b-thinking-2507`
- `stream`: `True` — odpowiedź strumieniowana token po tokenie
- temperatura domyślna modelu

**Ekstrakcja faktów**:
- `model`: `qwen/qwen3-4b-thinking-2507`
- `temperature`: `0.1` — niższa dla bardziej deterministycznych wyników
- `max_tokens`: `150` — ograniczenie długości odpowiedzi

---

## Prompt Engineering

### System prompt (generowanie odpowiedzi)

```
Jesteś pomocnym asystentem AI.
Odpowiadaj na pytania użytkownika,
korzystając z poniższego kontekstu i faktów o użytkowniku.

Kontekst z bazy wiedzy:
{rag_context}

Fakty o użytkowniku:
{memory_json}
```

### Prompt ekstrakcji faktów

```
Przeanalizuj ostatnią wymianę:

User: {prompt}
Assistant: {full_response}

Wyodrębnij nowe informacje o użytkowniku w formacie:
klucz: wartość
klucz: wartość

Jeśli nic nowego → napisz dokładnie: BRAK

Bez żadnego dodatkowego tekstu.
```

---

## Wydajność i ograniczenia

### Aktualne ograniczenia

| Obszar | Ograniczenie | Możliwe rozwiązanie |
|--------|-------------|---------------------|
| Chunking | Podział wyłącznie po liniach | Implementacja overlap-based chunking |
| Historia czatu | Brak limitu tokenów w historii | Truncation / summarization starszych wiadomości |
| Liczba wyników RAG | Sztywno ustawione `n_results=2` | Parametryzacja lub adaptive retrieval |
| Pamięć użytkownika | Brak walidacji ekstrahowanych faktów | Filtrowanie / schema validation |
| Duplikaty w ChromaDB | `ingest.py` dodaje duplikaty przy ponownym uruchomieniu | Sprawdzenie `collection.get()` przed dodaniem |

### Wydajność

- Czas odpowiedzi zależy od prędkości lokalnego GPU/CPU dla modeli LM Studio
- ChromaDB dla małych kolekcji (< 10k dokumentów) nie wymaga optymalizacji
- Streaming (`stream=True`) poprawia UX — użytkownik widzi odpowiedź w czasie rzeczywistym

---

## Bezpieczeństwo

- **Brak uwierzytelniania** — aplikacja dostępna dla wszystkich na lokalnym hoście
- **Brak sanityzacji wejścia** — prompty przekazywane bezpośrednio do LLM
- **Klucz API** — `api_key='lmstudio'` to placeholder, LM Studio nie wymaga prawdziwego klucza

Aplikacja przeznaczona jest wyłącznie do lokalnego użytku deweloperskiego/edukacyjnego.

---

## Rozszerzenia i możliwości rozwoju

- **Chunking z nakładką (overlap)** — lepsze pokrycie semantyczne przy podziale dokumentów
- **Re-ranking wyników RAG** — cross-encoder do poprawy trafności
- **Historia rozmów w bazie danych** — SQLite lub ChromaDB zamiast `session_state`
- **Upload dokumentów przez UI** — drag-and-drop plików PDF/TXT z auto-ingestion
- **Obsługa wielu użytkowników** — identyfikacja użytkownika i osobne pliki pamięci
- **Ewaluacja RAG** — metryki Recall@K, MRR dla oceny jakości wyszukiwania
