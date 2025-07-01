# Dokumentation: EWC-Integration in FastAPI-LoRA-Feintuning

Dieses Dokument beschreibt im Detail die Implementierungsschritte und enthält die wichtigsten Code-Beispiele zur Elastic Weight Consolidation (EWC).

---

## 1. Projektstruktur

```

project-root/server/
├── data/
│   └── corrections/
│       ├── audio_0001.wav
│       ├── text_0001.txt
│       └── …
└── models/
├── ewc-ckpt/
└── ewc-final/

```

---

## 2. FastAPI-Endpoints

### 2.1 `/upload/corrections`

Speichert Audio- und Textpaare:

```

from fastapi import FastAPI, UploadFile
import os, shutil

app = FastAPI()
CORR_DIR = "data/corrections"

@app.post("/upload/corrections")
async def upload_pair(audio: UploadFile, text: UploadFile):
idx = len(os.listdir(CORR_DIR)) // 2 + 1
wav_path = f"{CORR_DIR}/audio_{idx:04d}.wav"
txt_path = f"{CORR_DIR}/text_{idx:04d}.txt"
with open(wav_path, "wb") as f: shutil.copyfileobj(audio.file, f)
with open(txt_path, "wb") as f: shutil.copyfileobj(text.file, f)
return {"status": "saved", "id": idx}

```

### 2.2 `/train/ewc`

Startet das EWC-Feintuning als Hintergrund-Task:

```

from fastapi import BackgroundTasks

@app.post("/train/ewc")
def trigger_training(background_tasks: BackgroundTasks):
background_tasks.add_task(run_ewc_training)
return {"status": "training_started"}

```

---

## 3. Fisher-Information berechnen

Ermittelt die diagonale Fisher-Matrix und speichert alte Parameter:

```

import torch
from torch.autograd import Variable

def get_fisher_diag(model, dataloader):
model.eval()
params = {n: p for n, p in model.named_parameters() if p.requires_grad}
p_old = {n: Variable(p.data.clone()) for n, p in params.items()}
fisher = {n: torch.zeros_like(p) for n, p in params.items()}

    for batch in dataloader:
        inputs, labels = batch
        model.zero_grad()
        outputs = model(**inputs)
        loss = torch.nn.functional.nll_loss(
            torch.log_softmax(outputs.logits, dim=-1), labels
        )
        loss.backward()
        for n, p in model.named_parameters():
            if p.requires_grad:
                fisher[n] += p.grad.data.pow(2)
    
    for n in fisher:
        fisher[n] /= len(dataloader)
    return fisher, p_old
    ```

---

## 4. EWCTrainer mit EWC-Term

Erweitert den HuggingFace‐Trainer:

```

from transformers import Trainer

class EWCTrainer(Trainer):
def __init__(self, ewc_lambda, fisher, p_old, *args, **kwargs):
super().__init__(*args, **kwargs)
self.ewc_lambda = ewc_lambda
self.fisher = fisher
self.p_old = p_old

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        task_loss = outputs.loss if hasattr(outputs, "loss") else super().compute_loss(model, inputs)
        ewc_loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                ewc_loss += (self.fisher[n] * (p - self.p_old[n]).pow(2)).sum()
        loss = task_loss + (self.ewc_lambda / 2) * ewc_loss
        return (loss, outputs) if return_outputs else loss
    ```

---

## 5. Gesamtes Training: `run_ewc_training()`

```

import os
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor, TrainingArguments

class CorrectionDataset(Dataset):
def __init__(self, audio_paths, text_paths, processor):
self.audio_paths = audio_paths
self.text_paths = text_paths
self.processor = processor
def __len__(self): return len(self.audio_paths)
def __getitem__(self, i):
speech, _ = torchaudio.load(self.audio_paths[i])
text = open(self.text_paths[i]).read().strip()
inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
labels = self.processor.tokenizer(text, return_tensors="pt").input_ids
return {"input_values": inputs.input_values.squeeze(), "labels": labels.squeeze()}

def run_ewc_training():
model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")

    files = sorted(os.listdir("data/corrections"))
    audio_files = [f"data/corrections/{f}" for f in files if f.endswith(".wav")]
    text_files  = [f"data/corrections/{f}" for f in files if f.endswith(".txt")]
    
    ds_A = CorrectionDataset(audio_files, text_files, processor)
    ds_B = CorrectionDataset(audio_files, text_files, processor)
    dl_A = DataLoader(ds_A, batch_size=4, shuffle=True)
    dl_B = DataLoader(ds_B, batch_size=4, shuffle=True)
    
    fisher, p_old = get_fisher_diag(model, dl_A)
    
    training_args = TrainingArguments(
        output_dir="models/ewc-ckpt",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=50
    )
    
    trainer = EWCTrainer(
        ewc_lambda=100.0,
        fisher=fisher,
        p_old=p_old,
        model=model,
        args=training_args,
        train_dataset=ds_B
    )
    trainer.train()
    trainer.save_model("models/ewc-final")
    ```

---

