import numpy as np

class Collector(object):
    def __init__(self, filename, samples=54):
        self.file = open(filename, mode='w')
        self.file.write('dataset,')
        for i in range(samples):
            self.file.write(str(i / 54) + ',')
        self.file.write('\n')

    def get_roc(self, train, test, tumors):
        total = np.concatenate((train, test), axis=0)
        threshes = np.sort(total)
        positive_rates = []
        for t in threshes:
            positive_rates.append(np.count_nonzero(tumors >= t))
        return positive_rates

    def write_roc(self, dataset, rates):
        self.file.write(str(dataset) + ',')
        for r in rates:
            self.file.write(str(r) + ',')
        self.file.write('\n')

    def close(self):
        self.file.close()
