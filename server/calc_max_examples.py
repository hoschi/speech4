import argparse
import os

# Argumente parsen
parser = argparse.ArgumentParser(description='Berechne maximales N_VALIDATION für 13h Laufzeit bei allen Alpha/Beta-Kombinationen.')
parser.add_argument('-t', type=float, required=True, help='Zeit pro Kombination mit DEBUG=true Anzahl an Beispielen in Sekunden (z.B. 3908)')
args = parser.parse_args()

total_seconds = 13 * 60 * 60 # 13h

# DEBUG auf false setzen, damit constants.py die volle Range lädt
os.environ['DEBUG'] = 'false'

import constants

ALPHA_RANGE = constants.ALPHA_RANGE
beta_range = constants.beta_range

n_combinations = len(ALPHA_RANGE) * len(beta_range)

t = args.t
# Formel des Users:
# n_combinations * N_VALIDATION_max * t / EXAMPLES_COUNT_DEBUG <= total_seconds
# => N_VALIDATION_max = total_seconds / (n_combinations * t / EXAMPLES_COUNT_DEBUG)
N_VALIDATION_max = total_seconds / (n_combinations * t / constants.EXAMPLES_COUNT_DEBUG)

print(f"Alpha-Kombinationen: {len(ALPHA_RANGE)} {ALPHA_RANGE}")
print(f"Beta-Kombinationen: {len(beta_range)} {beta_range}")
print(f"Gesamt-Kombinationen: {n_combinations}")
print(f"Zeit pro Kombination (für {constants.EXAMPLES_COUNT_DEBUG} Beispiele): {t:.2f} Sekunden")
print(f"Maximal mögliches N_VALIDATION für 13h Laufzeit: {int(N_VALIDATION_max)}") 