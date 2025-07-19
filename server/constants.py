# Konstanten für das Decoder-Tuning und andere Server-Module

import os

EXAMPLES_COUNT_DEBUG = 350
EXAMPLES_COUNT_REAL = 800
DEBUG = os.environ.get('DEBUG', '').lower() == 'true'
N_VALIDATION = EXAMPLES_COUNT_DEBUG if DEBUG else EXAMPLES_COUNT_REAL

# Wertebereich für Alpha im Grid Search
ALPHA_RANGE = [round(x, 2) for x in list(__import__('numpy').linspace(0.3, 0.6, num=7))]
if DEBUG:
    ALPHA_RANGE = [ALPHA_RANGE[0]]

# Wertebereich für Beta im Grid Search
beta_range = [round(x, 2) for x in list(__import__('numpy').linspace(-1.5, -0.9, num=7))]
if DEBUG:
    beta_range = [beta_range[0]]
