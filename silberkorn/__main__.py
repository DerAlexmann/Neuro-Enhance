"""Start ueber  python -m silberkorn"""

import multiprocessing
import sys

from .start import main

# Prozesse wie der TensorRT-Bau importieren dieses Modul erneut - sie duerfen
# das Programm nicht noch einmal starten.
if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
