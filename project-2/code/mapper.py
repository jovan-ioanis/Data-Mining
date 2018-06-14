#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

def transform(x_original):
    return x_original

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  # Use our features.

