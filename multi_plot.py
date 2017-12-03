import roc_collection as roc
import matplotlib.pyplot as pplt
import glob

def by_iter():
    for i in range(3, 31, 3):
        # files = glob.glob('../longtest/step-lin-1')
        tab = roc.Tabulator('../longtest/step-lin-1-[' + str(i) + '].csv')
        pplt.plot(tab.points, tab.roc)
    pplt.show()


def by_step():
    for i in range(1, 4, 1):
        # files = glob.glob('../longtest/step-lin-1')
        tab = roc.Tabulator('../longtest/step-lin-' + str(i) + '-[27].csv')
        pplt.plot(tab.points, tab.roc)
    pplt.show()


if __name__ == '__main__':
    by_step()