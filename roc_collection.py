import numpy as np
import csv
import matplotlib.pyplot as mplot


class Collector(object):
    def __init__(self, filename, samples=54):
        self.file = open(filename, mode='w')
        self.file.write('dataset,')
        for i in range(samples):
            self.file.write(str(i / 54) + ',')
        self.file.write('\n')

    def get_roc(self, train, test, tumors):
        total = np.concatenate((train, test), axis=0)
        threshes = np.sort(total)[::-1]
        positive_rates = []
        for t in threshes:
            positive_rates.append(np.count_nonzero(tumors >= t) / 4)
        return positive_rates

    def write_roc(self, dataset, rates):
        self.file.write(str(dataset) + ',')
        for r in rates:
            self.file.write(str(r) + ',')
        self.file.write('\n')

    def close(self):
        self.file.close()


class Tabulator(object):
    def __init__(self, filename):
        self.set_roc = {}
        points = 0
        self.points = None
        with open(filename, mode='r') as data_file:
            reader = csv.DictReader(data_file)
            for row in reader:
                if points == 0:
                    points = len(row) - 2
                if self.points is None:
                    self.points = list(row.keys())[1:-1]
                index = int(row['dataset'])
                self.set_roc[index] = {}
                for k in list(row.keys())[1:-1]:
                #    if k == 'dataset':
                #        continue
                    self.set_roc[index][float(k)] = (float(row[k]))
        roc_m = np.ndarray((len(self.set_roc), points))
        for i, data in enumerate(self.set_roc.values()):
            for j, val in enumerate(sorted(data.keys())):
                roc_m[i][j] = data[val]
        self.roc = np.mean(roc_m, axis=0)

    def plot_roc(self):
        mplot.plot(self.points, self.roc)
        mplot.show()

    def auc(self):
        step = 1/len(self.points)
        return np.sum(step * self.roc)

    def __str__(self):
        return str(self.roc)

    def __repr__(self):
        return str(self.roc)


def convert_old_file(filename, suffix='-new', output=None):
    with open(filename, mode='r') as old:
        reader = csv.reader(old)
        header = reader.__next__()
        # header = [header[0], *header[:1:-1]]
        data = []
        indices = []
        for i, row in enumerate(reader):
            indices.append(row[0])
            data.append(list(map(float, row[1:-1])))
        data = np.asarray(data)
        data = 1 - data
        data = data[:, ::-1]
        new_name = filename[:-4] + suffix + '.csv' if output is None else output
        with open(new_name, mode='w') as new:
            new.write(header[0])
            for h in header[1:]:
                new.write(',' + h)
            new.write('\n')
            for j, index in enumerate(indices):
                new.write(index + ',')
                for val in data[j]:
                    new.write(str(val) + ',')
                new.write('\n')


if __name__ == '__main__':
    convert_old_file('first_test_tanh.csv')
    tab = Tabulator('first_test_tanh-new.csv')
    print(tab)
    print(len(tab.roc))
    tab.plot_roc()
    print(tab.auc())