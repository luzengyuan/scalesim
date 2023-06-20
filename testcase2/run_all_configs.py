import math

import multiprocessing as mp

from scalesim.topology_utils import topologies as topo
from scalesim.scale_config import scale_config as config
from scalesim.scale_sim import scalesim


def top_runner():
    topofilename = './files/tutorial1_topofile.csv'

    jobs = []

    for rpow in range(2, 11):
        rows = int(math.pow(2, rpow))
        cols = int( round(2 ** 12 / rows))

        for df in ['os', 'is', 'ws']:
            config_filename = './files/config/scale_config_' + str(rows) \
                              + 'x' + str(cols) + '_' + str(df) + '.cfg'

            j = mp.Process(target=run_scale, args=(config_filename, topofilename))
            jobs += [j]

    print('Number of jobs launched = ' + str(len(jobs)))

    for j in jobs:
        j.start()

    for j in jobs:
        j.join()


#
def run_scale(config_filename, topofilename):
    sim = scalesim( save_disk_space=True, verbose=False,
                    config=config_filename, topology=topofilename)

    sim.run_scale(top_path='tutorial1_runs')


if __name__ == '__main__':
    top_runner()