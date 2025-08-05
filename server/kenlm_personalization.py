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
    # YAML-Frontmatter am Anfang entfernen
    if text.startswith('---'):
        lines = text.splitlines()
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                end_idx = i
                break
        if end_idx is not None:
            text = '\n'.join(lines[end_idx+1:])
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
        self.model = kenlm.Model(str(self.model_path))
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
    def train_adaptive_pruning_pipeline(self, lambda_mix=0.95, regenerate_base_arpa=False):
        """
        Zweistufiges adaptives Pruning und Interpolation gemäß docs/2025-08-05-fix-fuer-kenlm-personalisierung-und-hotword-boosting.md
        1. Basiskorpus mit aggressivem Pruning trainieren
        2. Nutzerdaten mit mildem/ohne Pruning als separates Modell trainieren
        3. Interpolation/Merging der beiden ARPA-Modelle
        4. Finale ARPA als Binärmodell bauen
        """
        try:
            self.logger.info("Starte adaptive zweistufige Pruning-Pipeline...")
            # Schritt 1: Basiskorpus mit aggressivem Pruning (nur wenn Flag gesetzt)
            base_arpa = self.output_dir / "base_model.arpa"
            if regenerate_base_arpa or not base_arpa.exists():
                temp_dir = self.output_dir / "temp_base"
                temp_dir.mkdir(exist_ok=True)
                cmd_base = [
                    "kenlm/build/bin/lmplz",
                    "-o", "4",
                    "--prune", "0", "1", "1", "1",
                    "-S", "80%",
                    "-T", str(temp_dir),
                    "--text", str(self.base_corpus),
                    "--arpa", str(base_arpa),
                    "--verbose_header",
                    "--skip_symbols"
                ]
                self.logger.info(f"Basiskorpus: {' '.join(cmd_base)}")
                result_base = subprocess.run(cmd_base, capture_output=True, text=True)
                if result_base.returncode != 0:
                    self.logger.error(f"Basiskorpus-Training fehlgeschlagen: {result_base.stderr}")
                    raise RuntimeError("Basiskorpus-Training fehlgeschlagen")
            else:
                self.logger.info(f"Existierendes Basismodell (ARPA) wird verwendet: {base_arpa}")
            # Schritt 2: Nutzerdaten mit mildem/ohne Pruning
            corrections_cleaned, hotwords_path = self.preprocess_corrections()
            user_arpa = self.output_dir / "user_model.arpa"
            temp_dir_user = self.output_dir / "temp_user"
            temp_dir_user.mkdir(exist_ok=True)
            cmd_user = [
                "kenlm/build/bin/lmplz",
                "-o", "4",
                # Mildes Pruning, z.B. --prune 0 0 1
                "--prune", "0", "0", "1",
                "-S", "80%",
                "-T", str(temp_dir_user),
                "--text", str(corrections_cleaned),
                "--arpa", str(user_arpa),
                "--verbose_header",
                "--skip_symbols"
            ]
            self.logger.info(f"Nutzerdaten: {' '.join(cmd_user)}")
            result_user = subprocess.run(cmd_user, capture_output=True, text=True)
            # Prüfe auf BadDiscountException und retry mit --discount_fallback
            if result_user.returncode != 0:
                if "BadDiscountException" in result_user.stderr or "Could not calculate Kneser-Ney discounts" in result_user.stderr:
                    self.logger.warning("BadDiscountException erkannt, die Trainingsdaten sollten größer sein! Retry mit --discount_fallback...")
                    cmd_user_fallback = cmd_user + ["--discount_fallback"]
                    self.logger.info(f"Nutzerdaten (Fallback): {' '.join(cmd_user_fallback)}")
                    result_user_fallback = subprocess.run(cmd_user_fallback, capture_output=True, text=True)
                    if result_user_fallback.returncode != 0:
                        self.logger.error(f"Nutzerdaten-Training (Fallback) fehlgeschlagen: {result_user_fallback.stderr}")
                        raise RuntimeError("Nutzerdaten-Training fehlgeschlagen (Fallback)")
                else:
                    self.logger.error(f"Nutzerdaten-Training fehlgeschlagen: {result_user.stderr}")
                    raise RuntimeError("Nutzerdaten-Training fehlgeschlagen")
            # Schritt 3: Interpolation/Merging
            final_arpa = self.output_dir / "final_model.arpa"
            # SRILM ngram-Tool vorausgesetzt, alternativ eigenes Script
            cmd_interp = [
                "ngram",
                "-order", "4",
                "-lm", str(base_arpa),
                "-mix-lm", str(user_arpa),
                "-lambda", str(lambda_mix),
                "-write-lm", str(final_arpa)
            ]
            self.logger.info(f"Interpolation: {' '.join(cmd_interp)}")
            result_interp = subprocess.run(cmd_interp, capture_output=True, text=True)
            if result_interp.returncode != 0:
                self.logger.error(f"Interpolation fehlgeschlagen: {result_interp.stderr}")
                raise RuntimeError("Interpolation fehlgeschlagen")
            # Schritt 4: Binärmodell bauen
            binary_path = self.output_dir / "final_model.klm"
            cmd_bin = [
                "kenlm/build/bin/build_binary",
                "-a", "22",
                "-q", "8",
                "-b", "8",
                "trie",
                str(final_arpa),
                str(binary_path)
            ]
            self.logger.info(f"Binärmodell: {' '.join(cmd_bin)}")
            result_bin = subprocess.run(cmd_bin, capture_output=True, text=True)
            if result_bin.returncode != 0:
                self.logger.error(f"Binärkonvertierung fehlgeschlagen: {result_bin.stderr}")
                raise RuntimeError("Binärkonvertierung fehlgeschlagen")
            self.logger.info(f"Adaptive Pipeline erfolgreich! Modell: {binary_path}")
            return binary_path, hotwords_path
        except Exception as e:
            self.logger.error(f"Adaptive Pipeline fehlgeschlagen: {e}")
            raise
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
        hotwords_path = self.output_dir / "hotwords.txt"
        process_markdown_notes(self.user_correction_files, corrections_cleaned)
        # Hotwords extrahieren und speichern
        hotwords = self.extract_hotwords_from_corrections(self.user_correction_files)
        with open(hotwords_path, 'w', encoding='utf-8') as f:
            for word in hotwords:
                f.write(word + '\n')
        self.logger.info(f"Extrahierte Hotwords: {len(hotwords)} Begriffe")
        with open(corrections_cleaned, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.logger.info(f"Bereinigte Korrekturen: {len(lines)} Zeilen")
        return corrections_cleaned, hotwords_path

    @staticmethod
    def extract_hotwords_from_corrections(correction_files):
        hotwords = set()
        for filename in correction_files:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned = clean_markdown_for_kenlm(content)
                domain_terms = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{4,}\b', cleaned)
                hotwords.update(domain_terms)
                wiki_terms = re.findall(r'\[\[([^\]]+)\]\]', content)
                hotwords.update(wiki_terms)
        return list(hotwords)
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
    def train_complete_pipeline(self, lambda_mix=0.95):
        """
        Führt die komplette Pipeline immer mit zweistufigem adaptivem Pruning & Interpolation aus.
        """
        try:
            self.logger.info("Starte adaptive Pruning-Pipeline...")
            binary_path, hotwords_path = self.train_adaptive_pruning_pipeline(lambda_mix=lambda_mix)
            self.evaluate_model(binary_path)
            with open(hotwords_path, 'r', encoding='utf-8') as f:
                hotwords = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Adaptive Training erfolgreich abgeschlossen!")
            self.logger.info(f"Finales Modell: {binary_path}")
            return binary_path, hotwords
        except Exception as e:
            self.logger.error(f"Pipeline fehlgeschlagen: {e}")
            raise

