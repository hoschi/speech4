# Konstanten für das Decoder-Tuning und andere Server-Module

import os

DEBUG = os.environ.get('DEBUG', '').lower() == 'true'
N_VALIDATION = 350 if DEBUG else 336

# Wertebereich für Alpha im Grid Search
ALPHA_RANGE = [round(x, 2) for x in list(__import__('numpy').arange(0.0, 1, 0.2))]
if DEBUG:
    ALPHA_RANGE = [ALPHA_RANGE[0]]

# Wertebereich für Beta im Grid Search
beta_range = [round(x, 2) for x in list(__import__('numpy').arange(-2.0, 2.0, 0.4))]
if DEBUG:
    beta_range = [beta_range[0]]