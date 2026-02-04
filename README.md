# JAO Dashboard für Österreich

Dieses Repository enthält ein Streamlit-Dashboard zur Analyse des österreichischen Strommarkts (JAO/ENTSO-E).

## Voraussetzungen

- Python 3.9+

## Lokales Setup

### 1) Virtuelle Umgebung erstellen (empfohlen)

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

### 3) App starten

```bash
streamlit run app.py
```

Die App ist danach üblicherweise unter `http://localhost:8501` erreichbar.

## Hinweise

- In der Sidebar können Zeitraum, ENTSO-E API-Key und Länder ausgewählt werden.
- Daten werden erst nach Klick auf **Daten laden** geladen.
