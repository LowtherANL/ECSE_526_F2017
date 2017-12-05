import roc_collection as roc
import matplotlib.pyplot as pplt
import glob

def by_iter():
    iters = []
    for i in range(1, 15, 2):
        # files = glob.glob('../longtest/step-lin-1')
        tab = roc.Tabulator('../longtest/lin-step-lin-1-[' + str(i) + '].csv')
        pplt.plot(tab.points, tab.roc)
        iters.append(str(i)+ ' Learning Iterations')
    pplt.legend(iters)
    pplt.show()


def by_step():
    steps = []
    for i in range(1, 4, 1):
        # files = glob.glob('../longtest/step-lin-1')
        tab = roc.Tabulator('../longtest2/step-lin-' + str(i) + '-[27].csv')
        pplt.plot(tab.points, tab.roc)
        steps.append(str(i) + ' Steps in Middle Layer')
    pplt.legend(steps)
    pplt.show()


if __name__ == '__main__':
    by_step()
    #by_iter()
