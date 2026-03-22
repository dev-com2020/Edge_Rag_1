# Edge_Rag_1 — Lokalny Asystent AI z RAG i Pamięcią

Lokalny chatbot oparty na architekturze RAG (Retrieval-Augmented Generation) z persistentną pamięcią o użytkowniku. Działa w pełni offline, korzystając z LM Studio jako backendu LLM.

## Spis treści

- [Opis projektu](#opis-projektu)
- [Architektura systemu](#architektura-systemu)
- [Wymagania](#wymagania)
- [Instalacja i uruchomienie](#instalacja-i-uruchomienie)
- [Struktura plików](#struktura-plików)
- [Konfiguracja](#konfiguracja)
- [Jak działa RAG](#jak-działa-rag)
- [Pamięć użytkownika](#pamięć-użytkownika)
- [Baza wiedzy](#baza-wiedzy)
- [Rozwiązywanie problemów](#rozwiązywanie-problemów)

---

## Opis projektu

**Edge_Rag_1** to aplikacja webowa zbudowana w Streamlit, która łączy trzy kluczowe mechanizmy:

1. **RAG (Retrieval-Augmented Generation)** — przed odpowiedzią na pytanie system przeszukuje lokalną bazę wiedzy i dołącza trafne fragmenty do kontekstu modelu językowego.
2. **Wektorowa baza danych (ChromaDB)** — dokumenty z bazy wiedzy są indeksowane jako wektory semantyczne, co umożliwia wyszukiwanie po znaczeniu, nie tylko po słowach kluczowych.
3. **Persistentna pamięć użytkownika** — system automatycznie ekstrahuje fakty o użytkowniku z rozmów i zapamiętuje je między sesjami w pliku JSON.

Projekt działa **w pełni lokalnie** — nie wymaga połączenia z zewnętrznymi API.

---

## Architektura systemu

```
Użytkownik (przeglądarka)
        |
        v
  ┌─────────────┐
  │  Streamlit  │  ← interfejs czatu (app.py)
  └──────┬──────┘
         |
    Zapytanie użytkownika
         |
    ┌────┴────────────────────────┐
    |                             |
    v                             v
┌──────────┐              ┌──────────────┐
│ ChromaDB │              │  LM Studio   │
│ (RAG)    │              │  (LLM + EMB) │
└────┬─────┘              └──────┬───────┘
     |                           |
     | Trafne fragmenty          | Odpowiedź (streaming)
     └───────────┬───────────────┘
                 |
         ┌───────┴───────┐
         │  memory.json  │  ← fakty o użytkowniku
         └───────────────┘
```

**Przepływ danych:**

1. Użytkownik wpisuje pytanie
2. Pytanie jest zamieniane na embedding (wektor) przez model `nomic-embed-text`
3. ChromaDB wyszukuje 2 najdokładniej pasujące fragmenty wiedzy
4. LLM (`qwen3-4b`) generuje odpowiedź, mając dostęp do: fragmentów wiedzy + faktów o użytkowniku + pytania
5. Odpowiedź jest strumieniowana do interfejsu (token po tokenie)
6. LLM ekstrahuje nowe fakty o użytkowniku i zapisuje je w `memory.json`

---

## Wymagania

### Oprogramowanie

| Komponent | Wersja | Uwagi |
|-----------|--------|-------|
| Python | 3.10+ | |
| LM Studio | najnowsza | dostępny na [lmstudio.ai](https://lmstudio.ai) |
| Streamlit | 1.x | `pip install streamlit` |
| ChromaDB | 0.4+ | `pip install chromadb` |
| openai | 1.x | `pip install openai` |

### Modele w LM Studio

Przed uruchomieniem aplikacji pobierz i załaduj w LM Studio:

| Model | Zastosowanie | Identyfikator |
|-------|-------------|---------------|
| Qwen3 4B Thinking | generowanie odpowiedzi (LLM) | `qwen/qwen3-4b-thinking-2507` |
| Nomic Embed Text | generowanie embeddingów | `nomic-embed-text` |

LM Studio musi nasłuchiwać na domyślnym porcie: `http://localhost:1234/v1`

---

## Instalacja i uruchomienie

### 1. Klonowanie repozytorium

```bash
git clone <url-repozytorium>
cd Edge_Rag_1
```

### 2. Instalacja zależności

```bash
pip install streamlit chromadb openai
```

### 3. Uruchomienie LM Studio

1. Otwórz LM Studio
2. Pobierz modele: `qwen/qwen3-4b-thinking-2507` i `nomic-embed-text`
3. Uruchom serwer API (zakładka **Local Server**) na porcie `1234`
4. Załaduj oba modele

### 4. Zaindeksowanie bazy wiedzy (jednorazowo)

```bash
python ingest.py
```

Skrypt wczyta plik `data/wiedza.txt`, wygeneruje embeddingi i zapisze je w `./vector_store/`.

> Uwaga: wymagane jest, aby LM Studio z modelem embeddingowym było uruchomione.

### 5. Uruchomienie aplikacji

```bash
streamlit run app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`.

---

## Struktura plików

```
Edge_Rag_1/
├── app.py              # Główna aplikacja Streamlit (czat, RAG, pamięć)
├── ingest.py           # Skrypt indeksowania bazy wiedzy
├── memory.json         # Persistentna pamięć o użytkowniku (auto-generowany)
├── data/
│   └── wiedza.txt      # Baza wiedzy (dokumenty do wyszukiwania)
└── vector_store/       # ChromaDB — wektorowa baza danych (auto-generowany)
    └── ...
```

---

## Konfiguracja

Główne parametry konfiguracyjne znajdują się na początku pliku `app.py`:

```python
# URL serwera LM Studio
client = OpenAI(base_url="http://localhost:1234/v1", api_key='lmstudio')

# Model językowy do generowania odpowiedzi
LLM_MODEL = "qwen/qwen3-4b-thinking-2507"

# Model do generowania embeddingów (wektorów)
EMBEDDING_MODEL = "nomic-embed-text"

# Plik z pamięcią użytkownika
MEMORY_FILE = "memory.json"
```

Aby zmienić model lub adres serwera, zmodyfikuj powyższe zmienne.

---

## Jak działa RAG

RAG (Retrieval-Augmented Generation) rozwiązuje kluczowy problem modeli językowych — brak dostępu do aktualnej lub specyficznej wiedzy domenowej.

### Etap 1: Indeksowanie (ingest.py)

```
wiedza.txt → chunking (linie) → embeddingi → ChromaDB
```

- Dokument jest dzielony na linie (chunki)
- Każda linia jest zamieniana na wektor liczbowy (embedding) przez model semantyczny
- Wektory są przechowywane w ChromaDB z oryginalnym tekstem

### Etap 2: Wyszukiwanie (app.py)

```
Pytanie użytkownika → embedding zapytania → zapytanie do ChromaDB → top-2 fragmenty
```

- Pytanie użytkownika jest zamieniane na wektor
- ChromaDB oblicza podobieństwo kosinusowe między wektorem pytania a wszystkimi wektorami w bazie
- Zwracane są 2 najbardziej podobne fragmenty

### Etap 3: Generowanie odpowiedzi

```
System prompt = [fragmenty wiedzy] + [fakty o użytkowniku] + [pytanie] → LLM → odpowiedź
```

Model LLM ma dostęp do wyselekcjonowanego kontekstu, dzięki czemu odpowiada precyzyjnie na podstawie rzeczywistych danych.

---

## Pamięć użytkownika

System automatycznie uczy się faktów o użytkowniku w trakcie rozmowy.

### Mechanizm

Po każdej wymianie wiadomości LLM jest proszony o ekstrakcję nowych faktów w formacie:

```
klucz: wartość
klucz: wartość
```

Jeśli nie wykryto nowych faktów, model odpowiada `BRAK`.

### Przechowywanie

Fakty są zapisywane w pliku `memory.json`:

```json
{
  "imie": "Jan",
  "stanowisko": "developer",
  "preferowany_jezyk": "Python"
}
```

### Dostęp do pamięci w UI

Aktualna pamięć jest widoczna w panelu bocznym aplikacji (Streamlit sidebar).

### Czyszczenie pamięci

Aby zresetować pamięć użytkownika:

```bash
echo "{}" > memory.json
```

---

## Baza wiedzy

Baza wiedzy przechowywana jest w pliku `data/wiedza.txt`. Aktualnie zawiera **Politykę Pracy Zdalnej firmy AI-Corp**:

- Model hybrydowy — praca zdalna jako przywilej
- Wymagane dni w biurze: **wtorek i czwartek**
- Godziny pracy: **9:00 – 17:00 (czas polski)**
- Sprzęt firmowy: **laptop i monitor**
- Pracownik zapewnia własne łącze internetowe

### Dodawanie własnej wiedzy

1. Edytuj lub zastąp plik `data/wiedza.txt` własną treścią
2. Każda linia traktowana jest jako osobny chunk
3. Uruchom ponownie indeksowanie:

```bash
python ingest.py
```

> Przy ponownym uruchomieniu `ingest.py` dane zostaną nadpisane w kolekcji `rag_wiedza`.

---

## Rozwiązywanie problemów

### `chromadb.errors.InvalidCollectionException` — kolekcja nie istnieje

Uruchom najpierw skrypt indeksowania:

```bash
python ingest.py
```

### Brak odpowiedzi / timeout

Sprawdź, czy LM Studio działa i ma załadowane oba modele. Zweryfikuj dostępność:

```bash
curl http://localhost:1234/v1/models
```

### Model embeddingowy zwraca błąd

Upewnij się, że nazwa modelu w `ingest.py` (`EMBEDDING_MODEL`) zgadza się z nazwą załadowanego modelu w LM Studio. Identyfikatory mogą się różnić między wersjami.

### Pamięć zapisuje błędne dane

Jakość ekstrakcji faktów zależy od modelu LLM. W przypadku nieprawidłowych wpisów można ręcznie edytować `memory.json`.

---

## Licencja

Projekt edukacyjny. Brak ograniczeń licencyjnych.
