# Konstanten für das Decoder-Tuning und andere Server-Module

N_VALIDATION = 4 # 370

# Wertebereich für Alpha im Grid Search
ALPHA_RANGE = [0.2] # [round(x, 2) for x in list(__import__('numpy').arange(0.0, 1, 0.1))] 

# Wertebereich für Beta im Grid Search
beta_range = [-1.0] # [round(x, 2) for x in list(__import__('numpy').arange(-2.0, 2.1, 0.25))]