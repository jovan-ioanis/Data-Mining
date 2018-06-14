#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

if __name__ == "__main__":
    if not len(sys.argv) == 5:
        logging.error("Usage: evaluate.py weights.txt "
                      "test_data.txt test_labels.txt folder_with_mapper")
        sys.exit(1)

    # IMPORTANT: We must use the same feature transformation.
    sys.path.append(sys.argv[4])
    print sys.path
    from mapper import transform

    with open(sys.argv[1], "r") as fp_weights:
        weights = np.genfromtxt(fp_weights).flatten()

    accuracy = 0
    total = 0
    with open(sys.argv[2], "r") as fp_data:
        with open(sys.argv[3], "r") as fp_labels:
            for (x_string, label_string) in zip(fp_data, fp_labels):
                x_string = x_string.strip()
                label_string = label_string.strip()
                if not x_string:
                    assert not label_string
                    continue

                label = int(label_string)
                if label not in (-1, 1):
                    logging.error("Unknown label: %d" % label)
                    sys.exit(2)

                x_original = np.fromstring(x_string, sep=' ')

                # Transform the features.
                x = transform(x_original).flatten()  
                if not x.shape == weights.shape:
                    logging.error("Shapes of weight vector and transformed "
                                  "data don't match")
                    sys.exit(3)

                if np.abs(np.sum(weights)) < 1e-10:
                    logging.error("Zero vector provided")
                    sys.exit(4)

                if label*np.inner(weights, x) >= 0:
                    accuracy += 1
                total += 1

    print("%f" % (float(accuracy) / total))
