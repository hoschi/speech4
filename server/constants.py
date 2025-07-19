# Konstanten für das Decoder-Tuning und andere Server-Module

import os

EXAMPLES_COUNT_DEBUG = 350
EXAMPLES_COUNT_REAL = 800
DEBUG = os.environ.get('DEBUG', '').lower() == 'true'
N_VALIDATION = EXAMPLES_COUNT_DEBUG if DEBUG else EXAMPLES_COUNT_REAL

# Wertebereich für Alpha im Grid Search
ALPHA_RANGE = [round(x, 2) for x in list(__import__('numpy').arange(0.0, 1, 0.2))]
if DEBUG:
    ALPHA_RANGE = [ALPHA_RANGE[0]]

# Wertebereich für Beta im Grid Search
beta_range = [round(x, 2) for x in list(__import__('numpy').arange(-2.0, 2.0, 0.4))]
if DEBUG:
    beta_range = [beta_range[0]]
