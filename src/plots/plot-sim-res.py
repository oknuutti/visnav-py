import csv
import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    try:
        filename = sys.argv[1]
    except:
        print('USAGE: %s <logfile>' % sys.argv[0])
        quit()

    raw = []
    with open(filename, newline='') as fh:
        reader = csv.reader(fh, delimiter='\t', quotechar='"')
        for row in reader:
            raw.append(row)
    data = np.array(raw).astype('double')

    # time (1)
    # real D1 pose (7)
    # real D2 pose (7)          15
    # real s/c pose (7)
    # init s/c pose (7)         29
    # landmark algo loc (3)
    # landmark algo loc ok (1)  33
    # landmark algo ori (4)
    # landmark algo ori ok (1)  38
    # laser algo loc (3)
    # laser algo loc ok (1)     42
    # nav filter loc (3)
    # measurement log likelihood (1)
    # delta-v spent (1)
    fig, axs = plt.subplots(4, 1, sharex=True)

    # location errors
    axs[0].plot(data[:, 0]/3600, data[:, 15:18], ':')
    axs[0].set_prop_cycle(None)
    axs[0].plot(data[:, 0]/3600, data[:, 42:45])

    # measurements
    axs[1].plot(data[:, 0]/3600, data[:, 29:32], ':')
    axs[1].set_prop_cycle(None)
    axs[1].plot(data[:, 0]/3600, data[:, 38:41], '--')

    # measurement likelihood
    axs[2].plot(data[:, 0]/3600, data[:, 45])

    # delta-v used
    axs[3].plot(data[:, 0]/3600, data[:, 46])
    while not plt.waitforbuttonpress():
        pass




if __name__ == '__main__':
    main()