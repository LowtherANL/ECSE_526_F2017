import numpy as np
from roc_collection import Tabulator
import glob


def step_run_parser(filename):
    original = filename
    filename = filename[filename.rfind('/'):]
    filename = filename[filename.find('-') + 1:]
    filename = filename[filename.find('-') + 1:]
    step_start = filename.find('-')
    step_end = filename[step_start + 1:].find('-')
    step = int(filename[step_start + 1:step_start + 1 + step_end])
    bracket_start = filename.find('[')
    bracket_end = filename.find(']')
    iters = int(filename[bracket_start + 1:bracket_end])
    return original, step, iters


files = glob.glob('../longtest/*')
info = []
for file in files:
    info.append(step_run_parser(file))
sort_info = sorted(sorted(info, key=lambda t: t[2]), key=lambda t: t[1])
aucs = []
for f, s, i in sort_info:
    tab = Tabulator(f)
    aucs.append(tab.auc())
data = np.asarray(aucs)
data = data.reshape((2, 15))
print(data)