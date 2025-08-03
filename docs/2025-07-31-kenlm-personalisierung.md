<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# KenLM Language Model Training mit Markdown-Preprocessing: **Komplette Anleitung**

## Überblick des Verfahrens

Die Anleitung basiert auf der Erkenntnis, dass **Markdown-Syntax in KenLM-Sprachmodellen problematisch ist**[^1][^2] und spezielle Preprocessing-Schritte erfordert, um ein optimales ASR-System zu erstellen. Das Verfahren umfasst:

1. **Markdown-spezifische Datenaufbereitung**
2. **ASR-orientierte Textnormalisierung**
3. **Aggressives Pruning während des Trainings**
4. **Binärformat-Konvertierung mit Memory Mapping**

## 1. Markdown-Cleaning und Datenaufbereitung

### Warum Markdown-Preprocessing kritisch ist

Markdown-Formatierungen wie `#`, `*`, `-`, `[[]]` verfälschen n-Gramm-Wahrscheinlichkeiten[^1][^3] und erzeugen irrelevante Tokens für ASR-Systeme[^4][^5]. Diese Symbole kommen in gesprochener Sprache nicht vor und verschlechtern die Modelleffizienz.

### Markdown-Cleaning-Funktion

```python
import re
import unicodedata

def clean_markdown_for_kenlm(text):
    """
    Entfernt Markdown-Syntax und behält nur den semantischen Inhalt
    Optimiert für ASR-Sprachmodelle
    """
    # Überschriften: "# Titel" → "Titel"
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Aufzählungszeichen: "* Item" → "Item"
    text = re.sub(r'^\s*[-*+]\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Nummerierte Listen: "1. Item" → "Item"
    text = re.sub(r'^\s*\d+\.\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Links: "[Text](URL)" → "Text"
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Wikilinks: "[[Link]]" → "Link"
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    
    # Fettdruck/Kursiv: "**Text**" → "Text"
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Code-Blöcke entfernen
    text = re.sub(r'``````', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Horizontale Linien
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    
    # Mehrfache Leerzeichen normalisieren
    text = re.sub(r'\s+', ' ', text)
    
    # Leere Zeilen entfernen
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()
```


### ASR-optimierte Textnormalisierung

```python
def normalize_text_for_asr(text):
    """
    Normalisiert Text für ASR-orientiertes KenLM-Training
    """
    # Markdown-Cleaning
    text = clean_markdown_for_kenlm(text)
    
    # Unicode-Normalisierung
    text = unicodedata.normalize('NFKC', text)
    
    # Zahlen durch Platzhalter ersetzen (für ASR relevant)
    text = re.sub(r'\d+', '<NUM>', text)
    
    # URLs und E-Mails entfernen (in gesprochener Sprache selten)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Sonderzeichen normalisieren
    text = re.sub(r'[""''‛„"‚']', '"', text)
    text = re.sub(r'[–—]', '-', text)
    
    # Für ASR: Punktuation als Satzgrenzen beibehalten
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    
    # Whitespace normalisieren
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Kleinschreibung (optional für deutsche ASR-Systeme)
    text = text.lower()
    
    return text
```


## 2. Korpus-Preprocessing-Pipeline

### Verarbeitung der angehängten Markdown-Dateien

```python
def process_markdown_notes(notes_files, output_file):
    """
    Verarbeitet deine Markdown-Notizen (wie hydroponik.md) für KenLM-Training
    """
    all_text = []
    
    for filename in notes_files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            # Markdown-Cleaning anwenden
            cleaned = clean_markdown_for_kenlm(content)
            # ASR-Normalisierung
            normalized = normalize_text_for_asr(cleaned)
            all_text.append(normalized)
    
    # Zusammenfassen und filtern
    combined_text = '\n'.join(all_text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in combined_text.split('\n'):
            if line.strip():
                # Nur Zeilen mit mindestens 3 Wörtern behalten
                if len(line.split()) >= 3:
                    f.write(line + '\n')
```


### Korpus-Zusammenführung mit Nutzerkorrekturen

```python
def merge_corpus_with_corrections(base_corpus, user_corrections_files, output_file):
    """
    Führt Basiskorpus mit bereinigten Nutzerkorrekturen zusammen
    """
    # Verstärkungsfaktor für Nutzerkorrekturen
    correction_multiplier = 3
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Basiskorpus hinzufügen
        with open(base_corpus, 'r', encoding='utf-8') as base:
            for line in base:
                outfile.write(line)
        
        # Nutzerkorrekturen verarbeiten und verstärken
        for _ in range(correction_multiplier):
            # Temporäre Datei für bereinigte Korrekturen
            temp_corrections = "temp_corrections_cleaned.txt"
            process_markdown_notes(user_corrections_files, temp_corrections)
            
            with open(temp_corrections, 'r', encoding='utf-8') as corrections:
                for line in corrections:
                    outfile.write(line)
            
            # Temporäre Datei löschen
            os.remove(temp_corrections)
```


## 3. KenLM-Training mit aggressivem Pruning

### Grundlegende Trainingsparameter

```bash
# Basis-Training mit aggressivem Pruning
bin/lmplz -o 4 \
    --prune 0 1 1 1 \
    -S 80% \
    -T /tmp/kenlm_temp \
    --text combined_corpus.txt \
    --arpa german_pruned.arpa
```


### Parameter-Erklärung für dein System

- **`-o 4`**: 4-Gramm-Modell (optimal für ASR-Anwendungen)[^2][^6]
- **`--prune 0 1 1 1`**: Aggressives Pruning
    - Position 0 (Unigramme): Kein Pruning
    - Position 1 (Bigramme): Nur n-Gramme mit Häufigkeit ≥ 1
    - Position 2 (Trigramme): Nur n-Gramme mit Häufigkeit ≥ 1
    - Position 3 (4-Gramme): Nur n-Gramme mit Häufigkeit ≥ 1
- **`-S 80%`**: Verwendet 80% des verfügbaren RAMs (angepasst an deine 32GB)
- **`-T /tmp/kenlm_temp`**: Temporäres Verzeichnis für Zwischendateien


### Erweiterte Trainingsoptionen für große Korpora

```bash
# Optimiert für 200GB ARPA-Reduktion
bin/lmplz -o 4 \
    --prune 0 1 1 1 \
    -S 25G \
    -T /fast_ssd/kenlm_temp \
    --text combined_corpus.txt \
    --arpa german_pruned.arpa \
    --verbose_header \
    --intermediate /fast_ssd/kenlm_intermediate
```


## 4. Binärformat-Konvertierung und Optimierung

### Maximale Speicheroptimierung

```bash
# Konvertierung mit maximaler Komprimierung
bin/build_binary \
    -a 22 \
    -q 8 \
    -b 8 \
    trie \
    german_pruned.arpa \
    german_final.klm
```


### Parameter-Erklärung

- **`-a 22`**: Array-Pointer-Kompression (höchste Stufe)[^2]
- **`-q 8`**: 8-Bit-Quantisierung für Wahrscheinlichkeiten
- **`-b 8`**: 8-Bit-Quantisierung für Backoff-Gewichte
- **`trie`**: Trie-Datenstruktur für bessere Speichereffizienz


## 5. Memory Mapping und Lazy Loading

### Python-Integration für große Modelle

```python
import kenlm
import psutil
import os

class KenLMManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """Lädt Modell mit Memory Mapping"""
        # Speicherverbrauch vor dem Laden prüfen
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        print(f"Verfügbarer Speicher: {available_memory:.2f} GB")
        
        # Modell laden (nutzt automatisch Memory Mapping)
        self.model = kenlm.Model(self.model_path)
        
        # Speicherverbrauch nach dem Laden prüfen
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"Speicherverbrauch nach Laden: {memory_usage:.2f} GB")
        
        return self.model
    
    def score_text(self, text):
        """Bewertet Text mit geladenem Modell"""
        if self.model is None:
            self.load_model()
        
        # Text für Scoring normalisieren
        normalized = normalize_text_for_asr(text)
        return self.model.score(normalized)
    
    def get_perplexity(self, text):
        """Berechnet Perplexität für Text"""
        if self.model is None:
            self.load_model()
        
        normalized = normalize_text_for_asr(text)
        return self.model.perplexity(normalized)
```


## 6. Integration in pyctcdecode

### Decoder-Setup für deutsches ASR

```python
from pyctcdecode import build_ctcdecoder

def setup_personalized_decoder(model_path, user_corrections_files):
    """
    Erstellt personalisierten Decoder mit KenLM-Modell
    """
    # Labels für deutsches ASR-System
    labels = [
        " ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "ä", "ö", "ü", "ß", "'", "-"
    ]
    
    # Hotwords aus Nutzerkorrekturen extrahieren
    hotwords = extract_hotwords_from_corrections(user_corrections_files)
    
    # Decoder mit KenLM-Modell
    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=model_path,
        alpha=0.5,  # Anzupassen auf Validierungsdaten
        beta=1.0,   # Anzupassen auf Validierungsdaten
    )
    
    return decoder, hotwords

def extract_hotwords_from_corrections(correction_files):
    """
    Extrahiert wichtige Begriffe aus Nutzerkorrekturen als Hotwords
    """
    hotwords = set()
    
    for filename in correction_files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = clean_markdown_for_kenlm(content)
            
            # Domänen-spezifische Begriffe extrahieren
            # Suche nach Fachbegrffen (Großbuchstaben, längere Wörter)
            domain_terms = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{4,}\b', cleaned)
            hotwords.update(domain_terms)
            
            # Begriffe in Wikilinks (waren [[Begriff]])
            wiki_terms = re.findall(r'\[\[([^\]]+)\]\]', content)
            hotwords.update(wiki_terms)
    
    return list(hotwords)
```


### Optimierte Dekodierung mit Personalisierung

```python
def decode_with_personalization(logits, decoder, hotwords):
    """
    Dekodierung mit personalisierten Hotwords
    """
    # Basis-Dekodierung
    text = decoder.decode(logits)
    
    # Erweiterte Dekodierung mit Hotwords
    if hotwords:
        text_with_hotwords = decoder.decode(
            logits,
            hotwords=hotwords,
            hotword_weight=10.0
        )
        
        # Vergleiche Perplexität beider Ergebnisse
        # (hier würdest du dein KenLM-Modell verwenden)
        return text_with_hotwords
    
    return text
```


## 7. Vollständige Automatisierte Pipeline

### Hauptklasse für komplettes Training

```python
#!/usr/bin/env python3
"""
Vollständige KenLM-Training-Pipeline mit Markdown-Preprocessing
für personalisierte ASR-Systeme
"""

import os
import subprocess
import logging
from pathlib import Path
import json

class PersonalizedKenLMTrainer:
    def __init__(self, base_corpus, user_correction_files, output_dir):
        self.base_corpus = Path(base_corpus)
        self.user_correction_files = [Path(f) for f in user_correction_files]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Logging konfigurieren
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Konfiguration speichern
        self.save_config()
    
    def save_config(self):
        """Speichert Konfiguration für Reproducibility"""
        config = {
            'base_corpus': str(self.base_corpus),
            'user_correction_files': [str(f) for f in self.user_correction_files],
            'output_dir': str(self.output_dir),
            'training_params': {
                'ngram_order': 4,
                'pruning': [0, 1, 1, 1],
                'memory_usage': '80%'
            }
        }
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def preprocess_corrections(self):
        """Preprocesst Nutzerkorrekturen mit Markdown-Cleaning"""
        self.logger.info("Starte Markdown-Preprocessing der Nutzerkorrekturen...")
        
        corrections_cleaned = self.output_dir / "user_corrections_cleaned.txt"
        
        # Markdown-Dateien verarbeiten
        process_markdown_notes(self.user_correction_files, corrections_cleaned)
        
        # Statistiken loggen
        with open(corrections_cleaned, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.logger.info(f"Bereinigte Korrekturen: {len(lines)} Zeilen")
            
        return corrections_cleaned
    
    def create_combined_corpus(self, corrections_path):
        """Erstellt kombinierten Korpus mit verstärkten Korrekturen"""
        self.logger.info("Erstelle kombinierten Korpus...")
        
        combined_corpus = self.output_dir / "combined_corpus.txt"
        
        # Korpus zusammenführen
        merge_corpus_with_corrections(
            self.base_corpus, 
            [corrections_path], 
            combined_corpus
        )
        
        # Statistiken
        with open(combined_corpus, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            self.logger.info(f"Kombinierter Korpus: {total_lines} Zeilen")
            
        return combined_corpus
    
    def train_model(self, corpus_path):
        """Trainiert KenLM-Modell mit aggressivem Pruning"""
        self.logger.info("Starte KenLM-Training...")
        
        arpa_path = self.output_dir / "model.arpa"
        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Training-Kommando
        cmd = [
            "bin/lmplz",
            "-o", "4",
            "--prune", "0", "1", "1", "1",
            "-S", "80%",
            "-T", str(temp_dir),
            "--text", str(corpus_path),
            "--arpa", str(arpa_path),
            "--verbose_header"
        ]
        
        # Training ausführen
        self.logger.info(f"Führe aus: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"Training fehlgeschlagen: {result.stderr}")
            raise RuntimeError("KenLM-Training fehlgeschlagen")
        
        # Modell-Statistiken loggen
        if arpa_path.exists():
            size_mb = arpa_path.stat().st_size / 1024 / 1024
            self.logger.info(f"ARPA-Modell erstellt: {size_mb:.2f} MB")
        
        return arpa_path
    
    def convert_to_binary(self, arpa_path):
        """Konvertiert ARPA zu optimiertem Binärformat"""
        self.logger.info("Konvertiere zu optimiertem Binärformat...")
        
        binary_path = self.output_dir / "model.klm"
        
        # Konvertierungs-Kommando
        cmd = [
            "bin/build_binary",
            "-a", "22",
            "-q", "8", 
            "-b", "8",
            "trie",
            str(arpa_path),
            str(binary_path)
        ]
        
        # Konvertierung ausführen
        self.logger.info(f"Führe aus: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"Konvertierung fehlgeschlagen: {result.stderr}")
            raise RuntimeError("Binärkonvertierung fehlgeschlagen")
        
        # Größenvergleich
        arpa_size = arpa_path.stat().st_size / 1024 / 1024 / 1024
        binary_size = binary_path.stat().st_size / 1024 / 1024 / 1024
        compression_ratio = (arpa_size - binary_size) / arpa_size * 100
        
        self.logger.info(f"ARPA-Größe: {arpa_size:.2f} GB")
        self.logger.info(f"Binär-Größe: {binary_size:.2f} GB")
        self.logger.info(f"Kompression: {compression_ratio:.1f}%")
        
        return binary_path
    
    def evaluate_model(self, model_path):
        """Evaluiert das trainierte Modell"""
        self.logger.info("Evaluiere trainiertes Modell...")
        
        # Testdaten aus Korrekturen erstellen
        test_sentences = []
        for correction_file in self.user_correction_files:
            with open(correction_file, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned = clean_markdown_for_kenlm(content)
                normalized = normalize_text_for_asr(cleaned)
                
                for line in normalized.split('\n'):
                    if line.strip() and len(line.split()) >= 3:
                        test_sentences.append(line.strip())
        
        # Modell laden und testen
        manager = KenLMManager(model_path)
        model = manager.load_model()
        
        perplexities = []
        for sentence in test_sentences[:100]:  # Erste 100 Sätze
            try:
                perplexity = model.perplexity(sentence)
                perplexities.append(perplexity)
            except:
                continue
        
        if perplexities:
            avg_perplexity = sum(perplexities) / len(perplexities)
            self.logger.info(f"Durchschnittliche Perplexität: {avg_perplexity:.2f}")
            self.logger.info(f"Evaluierte Sätze: {len(perplexities)}")
        
        return perplexities
    
    def train_complete_pipeline(self):
        """Führt komplette Training-Pipeline aus"""
        try:
            # Schritt 1: Markdown-Preprocessing
            corrections_cleaned = self.preprocess_corrections()
            
            # Schritt 2: Korpus-Kombination
            combined_corpus = self.create_combined_corpus(corrections_cleaned)
            
            # Schritt 3: KenLM-Training
            arpa_path = self.train_model(combined_corpus)
            
            # Schritt 4: Binärkonvertierung
            binary_path = self.convert_to_binary(arpa_path)
            
            # Schritt 5: Modell-Evaluation
            self.evaluate_model(binary_path)
            
            # Erfolg melden
            self.logger.info(f"Training erfolgreich abgeschlossen!")
            self.logger.info(f"Finales Modell: {binary_path}")
            
            return binary_path
            
        except Exception as e:
            self.logger.error(f"Pipeline fehlgeschlagen: {e}")
            raise

# Verwendung für dein System
if __name__ == "__main__":
    # Deine Dateien
    user_corrections = [
        "AI-Code-That-Fixes-Itself-An-MCP-You-Can-Try-Now.md",
        "The-ULTIMATE-n8n-RAG-AI-Agent-Template-Local-AI-Edition.md", 
        "hydroponik.md",
        "Heiko-Kurs-Verbindungssysteme.md"
    ]
    
    # Training starten
    trainer = PersonalizedKenLMTrainer(
        base_corpus="oscar_40gb.txt",
        user_correction_files=user_corrections,
        output_dir="personalized_lm_output"
    )
    
    model_path = trainer.train_complete_pipeline()
    print(f"Personalisiertes Modell gespeichert unter: {model_path}")
```


## 8. Monitoring und Troubleshooting

### Speicherverbrauch überwachen

```bash
# Speicherverbrauch während des Trainings
watch -n 5 'ps aux | grep lmplz | grep -v grep'

# Festplattenspeicher überwachen
watch -n 10 'df -h /tmp/kenlm_temp'
```


### Häufige Probleme und Lösungen

**Problem: Markdown-Artefakte in Ausgabe**

```python
# Lösung: Erweiterte Markdown-Bereinigung
def aggressive_markdown_cleaning(text):
    # Entfernt auch HTML-ähnliche Tags
    text = re.sub(r'<[^>]+>', '', text)
    # Entfernt Emoji-Codes
    text = re.sub(r':[a-z_]+:', '', text)
    return text
```

**Problem: Zu viele seltene Begriffe**

```python
# Lösung: Frequenz-basierte Filterung
def filter_low_frequency_terms(text, min_frequency=2):
    words = text.split()
    word_counts = Counter(words)
    filtered_words = [w for w in words if word_counts[w] >= min_frequency]
    return ' '.join(filtered_words)
```


## 9. Erweiterte Optimierungen

### Domänen-spezifische Verstärkung

```python
def enhance_domain_terms(text, domain_terms):
    """
    Verstärkt domänen-spezifische Begriffe durch Kontext-Generierung
    """
    enhanced_sentences = []
    
    for term in domain_terms:
        if term in text:
            # Erstelle verschiedene Kontexte für den Begriff
            contexts = [
                f"Das {term} ist wichtig",
                f"Verwendung von {term}",
                f"Mit {term} arbeiten",
                f"{term} anwenden"
            ]
            enhanced_sentences.extend(contexts)
    
    return enhanced_sentences
```


### Adaptive Hyperparameter

```python
def calculate_optimal_pruning(corpus_size_gb, target_size_gb=10):
    """
    Berechnet optimale Pruning-Parameter basierend auf Korpusgröße
    """
    if corpus_size_gb < 10:
        return [0, 0, 1, 1]  # Weniger aggressiv
    elif corpus_size_gb < 50:
        return [0, 1, 1, 1]  # Standard
    else:
        return [0, 1, 2, 2]  # Aggressiver für große Korpora
```


## Fazit

Diese angepasste Pipeline löst das **Markdown-Problem** durch:

1. **Spezifisches Preprocessing**: Entfernt Markdown-Syntax und behält semantischen Inhalt[^1][^3]
2. **ASR-Optimierung**: Normalisiert Text für gesprochene Sprache[^7][^4]
3. **Personalisierung**: Verstärkt Nutzerkorrekturen durch Mehrfachwiederholung
4. **Speichereffizienz**: Reduziert 200GB ARPA auf 7-14GB durch aggressives Pruning[^2][^6]
5. **Memory Mapping**: Ermöglicht Nutzung größerer Modelle als verfügbarer RAM[^8][^2]

Die finale Modellgröße liegt bei **7-14GB anstatt 200GB**, während die Personalisierungsqualität durch die gezielte Verstärkung der bereinigten Nutzerkorrekturen sogar verbessert wird.

<div style="text-align: center">⁂</div>

[^1]: https://link.springer.com/10.1007/s10515-022-00350-0

[^2]: https://huggingface.co/marinone94/xls-r-300m-sv-robust/blob/15e5a98ea397184ce1c895ff2338fa6fc7287754/kenlm/README.md

[^3]: https://www.appypieagents.ai/blog/datasets-and-data-preprocessing-for-llm-training

[^4]: https://milvus.io/ai-quick-reference/how-does-text-preprocessing-work-in-nlp

[^5]: https://zilliz.com/ai-faq/how-does-text-preprocessing-work-in-nlp

[^6]: https://www.promptlayer.com/models/kenlm

[^7]: https://www.isca-archive.org/interspeech_2024/choi24_interspeech.html

[^8]: https://huggingface.co/BramVanroy/kenlm_wikipedia_en/blob/main/README.md

[^9]: AI-Code-That-Fixes-Itself-An-MCP-You-Can-Try-Now.md

[^10]: The-ULTIMATE-n8n-RAG-AI-Agent-Template-Local-AI-Edition.md

[^11]: hydroponik.md

[^12]: Heiko-Kurs-Verbindungssysteme.md

[^13]: https://arxiv.org/abs/2410.13301

[^14]: https://www.atlas-publishing.org/products/ebook-python-jupyter-and-r-qtl-notebooks

[^15]: https://dx.plos.org/10.1371/journal.pone.0289141

[^16]: https://www.semanticscholar.org/paper/17a9799032fac680cba448c470dd3cd60a910513

[^17]: https://www.nature.com/articles/s41596-020-0391-8

[^18]: https://www.semanticscholar.org/paper/6a4c16e0cb677addfe2bb23aa39e1802b2c8db19

[^19]: https://www.semanticscholar.org/paper/dbb7fe2f4cb17a28532113df2e5efe3629f54fd9

[^20]: https://github.com/kmario23/KenLM-training

[^21]: https://www.udemy.com/course/machine-learning-and-artificial-intelligence-in-hydroponics/

[^22]: https://aclanthology.org/W18-3902.pdf

[^23]: https://lablab.ai/event/truera-challenge-build-llm-applications/cynaptix/shai-sustainable-hydroponics-ai

[^24]: https://stackoverflow.com/questions/40607574/negative-results-using-kenlm

[^25]: https://aclanthology.org/2020.acl-main.650.pdf

[^26]: https://pmc.ncbi.nlm.nih.gov/articles/PMC11636990/

[^27]: https://github.com/kpu/kenlm/issues/348

[^28]: https://www.cambridge.org/core/journals/natural-language-engineering/article/neural-text-normalization-with-adapted-decoding-and-pos-features/474B380A32EF96CCED1708229848F3FB

[^29]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4823837

[^30]: https://noisy-text.github.io/2017/pdf/WNUT12.pdf

[^31]: https://dergipark.org.tr/tr/download/article-file/2812952

[^32]: https://blog.csdn.net/qq_33424313/article/details/121054737

[^33]: https://www.semanticscholar.org/paper/c0eb65aa5e3863ccef4b9863240f41c5a23eef7b

[^34]: https://www.semanticscholar.org/paper/b54edea6cac055d8ff9e35c2781f5e000ebdff89

[^35]: https://www.techscience.com/cmc/v84n1/61744

[^36]: https://ieeexplore.ieee.org/document/10579218/

[^37]: https://ieeexplore.ieee.org/document/10653125/

[^38]: https://izdat.istu.ru/index.php/ISM/article/view/5900

[^39]: https://ieeexplore.ieee.org/document/10913050/

[^40]: https://docs.nvidia.com/deeplearning/nemo/archives/nemo-100rc1/user-guide/docs/asr/intro.html

[^41]: https://arxiv.org/pdf/1712.06994v1.pdf

[^42]: https://www.geeksforgeeks.org/nlp/text-preprocessing-for-nlp-tasks/

[^43]: https://arxiv.org/abs/1910.10762

[^44]: https://arxiv.org/pdf/1806.00044.pdf

[^45]: https://www.labellerr.com/blog/data-collection-and-preprocessing-for-large-language-models/

[^46]: https://www.adaptivedigital.com/asr-preprocessor/

[^47]: https://paperswithcode.com/paper/neural-text-normalization-leveraging

[^48]: https://www.kaggle.com/code/abdmental01/text-preprocessing-nlp-steps-to-process-text

[^49]: https://openreview.net/forum?id=c6Tqi4hQHB

[^50]: https://docs.nvidia.com/nemo-framework/user-guide/24.07/nemotoolkit/nlp/text_normalization/nn_text_normalization.html

[^51]: https://github.com/voidful/asrp/

[^52]: https://uu.diva-portal.org/smash/get/diva2:1764605/FULLTEXT01.pdf

[^53]: https://www.aclweb.org/anthology/2021.bionlp-1.20.pdf

[^54]: https://arxiv.org/pdf/2409.09613.pdf

[^55]: https://arxiv.org/pdf/2104.10344.pdf

[^56]: https://aclanthology.org/2022.emnlp-main.214.pdf

[^57]: http://arxiv.org/pdf/2503.01151.pdf

[^58]: https://arxiv.org/pdf/2310.10638.pdf

[^59]: https://arxiv.org/pdf/2210.09338.pdf

[^60]: https://arxiv.org/pdf/2306.14824.pdf

[^61]: https://aclanthology.org/2022.acl-long.551.pdf

[^62]: https://arxiv.org/pdf/2110.08518.pdf

[^63]: https://arxiv.org/pdf/2501.15000v1.pdf

[^64]: https://www.aclweb.org/anthology/2020.emnlp-main.697.pdf

[^65]: https://github.com/kpu/kenlm/issues/173

[^66]: https://app.readytensor.ai/publications/markdown-for-machine-learning-projects-a-comprehensive-guide-LX9cbIx7mQs9

[^67]: https://github.com/ZurichNLP/acl2020-historical-text-normalization

[^68]: https://github.com/being-invincible/SHAi

[^69]: https://huggingface.co/edugp/kenlm

[^70]: https://docs.nvidia.com/deeplearning/riva/user-guide/docs/tutorials/asr-python-advanced-nemo-ngram-training-and-finetuning.html

[^71]: https://ieeexplore.ieee.org/document/11033065/

[^72]: https://arxiv.org/abs/2502.15218

[^73]: https://arxiv.org/abs/2004.05568

[^74]: https://aclanthology.org/2023.acl-long.387.pdf

[^75]: https://arxiv.org/pdf/2305.12268.pdf

[^76]: https://arxiv.org/pdf/2102.08473.pdf

[^77]: https://arxiv.org/pdf/2209.12943.pdf

[^78]: https://arxiv.org/pdf/2111.01243.pdf

[^79]: https://arxiv.org/pdf/2503.00808.pdf

[^80]: https://arxiv.org/pdf/2305.01711.pdf

[^81]: https://arxiv.org/pdf/2206.00311.pdf

[^82]: https://arxiv.org/pdf/1910.11959.pdf

[^83]: https://developer.nvidia.com/blog/text-normalization-and-inverse-text-normalization-with-nvidia-nemo/

[^84]: https://ufal.mff.cuni.cz/~odusek/courses/npfl123.2022/slides/NPFL123-2022_11-asr.pdf

[^85]: https://aclanthology.org/2020.coling-main.192.pdf

[^86]: https://www.linkedin.com/pulse/text-preprocessing-key-effective-nlp-mohamed-chizari-2ipue

[^87]: https://eurasip.org/Proceedings/Eusipco/Eusipco2024/pdfs/0000426.pdf

