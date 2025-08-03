"""
KenLM-Personalisierung für ASR-Systeme
Basierend auf der Anleitung in docs/2025-07-31-kenlm-personalisierung.md
"""
import re
import unicodedata
import os
import subprocess
import logging
from pathlib import Path
import json
import kenlm
import psutil

# Markdown-Cleaning

def clean_markdown_for_kenlm(text):
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'``````', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^---+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

# ASR-Normalisierung

def normalize_text_for_asr(text):
    text = clean_markdown_for_kenlm(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'["‟‛„‚]', '"', text)
    text = re.sub(r'[–—]', '-', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = text.lower()
    return text

# Markdown-Notizen verarbeiten

def process_markdown_notes(notes_files, output_file):
    all_text = []
    for filename in notes_files:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = clean_markdown_for_kenlm(content)
            normalized = normalize_text_for_asr(cleaned)
            all_text.append(normalized)
    combined_text = '\n'.join(all_text)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in combined_text.split('\n'):
            if line.strip() and len(line.split()) >= 3:
                f.write(line + '\n')

# Korpus-Zusammenführung

def merge_corpus_with_corrections(base_corpus, user_corrections_files, output_file):
    correction_multiplier = 3
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(base_corpus, 'r', encoding='utf-8') as base:
            for line in base:
                outfile.write(line)
        for _ in range(correction_multiplier):
            temp_corrections = "temp_corrections_cleaned.txt"
            process_markdown_notes(user_corrections_files, temp_corrections)
            with open(temp_corrections, 'r', encoding='utf-8') as corrections:
                for line in corrections:
                    outfile.write(line)
            os.remove(temp_corrections)

# KenLM-Manager

class KenLMManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
    def load_model(self):
        available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024
        print(f"Verfügbarer Speicher: {available_memory:.2f} GB")
        self.model = kenlm.Model(self.model_path)
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"Speicherverbrauch nach Laden: {memory_usage:.2f} GB")
        return self.model
    def score_text(self, text):
        if self.model is None:
            self.load_model()
        normalized = normalize_text_for_asr(text)
        return self.model.score(normalized)
    def get_perplexity(self, text):
        if self.model is None:
            self.load_model()
        normalized = normalize_text_for_asr(text)
        return self.model.perplexity(normalized)

# Hauptklasse für Training

class PersonalizedKenLMTrainer:
    def __init__(self, base_corpus, user_correction_files, output_dir):
        self.base_corpus = Path(base_corpus)
        self.user_correction_files = [Path(f) for f in user_correction_files]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Logging in server/reports/training
        reports_dir = Path("server/reports/training")
        reports_dir.mkdir(parents=True, exist_ok=True)
        # Commit-Hash für eindeutige Zuordnung
        from utils import get_current_commit_hash
        commit_hash = get_current_commit_hash()
        log_path = reports_dir / f"training_{self.output_dir.name}_{commit_hash}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.save_config()
    def save_config(self):
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
        self.logger.info("Starte Markdown-Preprocessing der Nutzerkorrekturen...")
        corrections_cleaned = self.output_dir / "user_corrections_cleaned.txt"
        process_markdown_notes(self.user_correction_files, corrections_cleaned)
        with open(corrections_cleaned, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.logger.info(f"Bereinigte Korrekturen: {len(lines)} Zeilen")
        return corrections_cleaned
    def create_combined_corpus(self, corrections_path):
        self.logger.info("Erstelle kombinierten Korpus...")
        combined_corpus = self.output_dir / "combined_corpus.txt"
        merge_corpus_with_corrections(
            self.base_corpus,
            [corrections_path],
            combined_corpus
        )
        with open(combined_corpus, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            self.logger.info(f"Kombinierter Korpus: {total_lines} Zeilen")
        return combined_corpus
    def train_model(self, corpus_path):
        self.logger.info("Starte KenLM-Training...")
        arpa_path = self.output_dir / "model.arpa"
        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        cmd = [
            "kenlm/build/bin/lmplz",
            "-o", "4",
            "--prune", "0", "1", "1", "1",
            "-S", "80%",
            "-T", str(temp_dir),
            "--text", str(corpus_path),
            "--arpa", str(arpa_path),
            "--verbose_header"
        ]
        self.logger.info(f"Führe aus: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Training fehlgeschlagen: {result.stderr}")
            raise RuntimeError("KenLM-Training fehlgeschlagen")
        if arpa_path.exists():
            size_mb = arpa_path.stat().st_size / 1024 / 1024
            self.logger.info(f"ARPA-Modell erstellt: {size_mb:.2f} MB")
        return arpa_path
    def convert_to_binary(self, arpa_path):
        self.logger.info("Konvertiere zu optimiertem Binärformat...")
        binary_path = self.output_dir / "model.klm"
        cmd = [
            "kenlm/build/bin/build_binary",
            "-a", "22",
            "-q", "8",
            "-b", "8",
            "trie",
            str(arpa_path),
            str(binary_path)
        ]
        self.logger.info(f"Führe aus: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            self.logger.error(f"Konvertierung fehlgeschlagen: {result.stderr}")
            raise RuntimeError("Binärkonvertierung fehlgeschlagen")
        arpa_size = arpa_path.stat().st_size / 1024 / 1024 / 1024
        binary_size = binary_path.stat().st_size / 1024 / 1024 / 1024
        compression_ratio = (arpa_size - binary_size) / arpa_size * 100
        self.logger.info(f"ARPA-Größe: {arpa_size:.2f} GB")
        self.logger.info(f"Binär-Größe: {binary_size:.2f} GB")
        self.logger.info(f"Kompression: {compression_ratio:.1f}%")
        return binary_path
    def evaluate_model(self, model_path):
        self.logger.info("Evaluiere trainiertes Modell...")
        test_sentences = []
        for correction_file in self.user_correction_files:
            with open(correction_file, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned = clean_markdown_for_kenlm(content)
                normalized = normalize_text_for_asr(cleaned)
                for line in normalized.split('\n'):
                    if line.strip() and len(line.split()) >= 3:
                        test_sentences.append(line.strip())
        manager = KenLMManager(model_path)
        model = manager.load_model()
        perplexities = []
        for sentence in test_sentences[:100]:
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
        try:
            corrections_cleaned = self.preprocess_corrections()
            combined_corpus = self.create_combined_corpus(corrections_cleaned)
            arpa_path = self.train_model(combined_corpus)
            binary_path = self.convert_to_binary(arpa_path)
            self.evaluate_model(binary_path)
            self.logger.info(f"Training erfolgreich abgeschlossen!")
            self.logger.info(f"Finales Modell: {binary_path}")
            return binary_path
        except Exception as e:
            self.logger.error(f"Pipeline fehlgeschlagen: {e}")
            raise

