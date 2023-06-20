import math
import numpy as np
import matplotlib.pyplot as plt

from scalesim.scale_sim import scalesim
from scalesim.utilities.scalesim_report import ScalesimReport as reporter


def run_scale_sim():
    # 1. Read the topology file
    topofilename = './files/tutorial1_topofile.csv'

    # 2. Read the config file
    config_filename = './files/config/scale_config_32x128_ws.cfg'

    # 3. Launch a SCALE-Sim run
    sim = scalesim( save_disk_space=True, verbose=True,
                    config=config_filename, topology=topofilename
        )

    sim.run_scale(top_path='tutorial1_runs')


# 4. Do the same for the other two dataflows
def compare_dataflows():
    run_dir = './tutorial1_runs'
    os_run_name = 'scale_sim_tutorial1_64x64_os'
    ws_run_name = 'scale_sim_tutorial1_64x64_ws'
    is_run_name = 'scale_sim_tutorial1_64x64_is'

    os_rpt = reporter()
    ws_rpt = reporter()
    is_rpt = reporter()

    os_rpt.load_data(data_dir=run_dir, run_name=os_run_name)
    ws_rpt.load_data(data_dir=run_dir, run_name=ws_run_name)
    is_rpt.load_data(data_dir=run_dir, run_name=is_run_name)

    # Prepare arrays for plotting
    x_tick_lables = ['64x64_OS', '64x64_WS', '64x64_IS']
    y_legend = ['Layer0', 'Layer1', 'Layer2']

    os_runtimes = os_rpt.get_compute_cycles_all_layer()
    ws_runtimes = ws_rpt.get_compute_cycles_all_layer()
    is_runtimes = is_rpt.get_compute_cycles_all_layer()

    sram_reads = os_rpt.get_sram_ifmap_reads_single_layer(0)

    # Transpose y data
    all_y = np.asarray(os_runtimes).reshape((1,3))
    all_y = np.concatenate((all_y, np.asarray(ws_runtimes).reshape((1,3))), axis=0)
    all_y = np.concatenate((all_y, np.asarray(is_runtimes).reshape((1,3))), axis=0)

    y_series = np.transpose(all_y)

    plot_stacked_bar(x=x_tick_lables, y_series_np=y_series, legends=y_legend, title='Runtime vs dataflow',
                     y_axis_label='Runtime (Cycles)')


# 5. Compare the aspect ratios and dataflows
def compare_aspect_ratios():
    run_dir = './tutorial1_runs'
    x_tick_labels = []
    all_y = np.zeros((1,1))
    data_valid = False
    y_legend = ['Layer0', 'Layer1', 'Layer2']

    for rpow in range(2, 11):
        rows = int(math.pow(2, rpow))
        cols = int(round(2 ** 12 / rows))

        for df in ['os', 'is', 'ws']:
            run_name = 'scale_sim_tutorial1_' + str(rows) + 'x' + str(cols) + '_' + str(df)
            x_tick_labels += [str(rows) + 'x' + str(cols) + '_' + str(df)]
            rpt = reporter()
            rpt.load_data(data_dir=run_dir, run_name=run_name)
            runtimes = rpt.get_compute_cycles_all_layer()

            if not data_valid:
                all_y = np.asarray(runtimes).reshape((1,3))
                data_valid = True
            else:
                all_y = np.concatenate((all_y, np.asarray(runtimes).reshape((1, 3))), axis=0)

    y_series = np.transpose(all_y)

    plot_stacked_bar(x=x_tick_labels, y_series_np=y_series, legends=y_legend, title='Runtime vs dataflow',
                     y_axis_label='Runtime (Cycles)')


#
def plot_stacked_bar(x, y_series_np, legends, title, y_axis_label=''):
    num_plots = y_series_np.shape[0]
    plt.bar(x, y_series_np[0], label=legends[0])
    bottom = y_series_np[0]
    for plt_id in range(1, num_plots):
        plt.bar(x, y_series_np[plt_id], bottom=bottom,label=legends[plt_id])
        bottom += y_series_np[plt_id]

    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.xticks(rotation=70)
    plt.legend()

    plt.show()



if __name__ == '__main__':
    #run_scale_sim()
    #compare_dataflows()
    compare_aspect_ratios()
