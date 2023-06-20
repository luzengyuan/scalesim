import os
import argparse


from scale_config import scale_config
#from scalesim.topology_utils import topologies
from topology_utils import topologies
from simulator import simulator as sim


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', metavar='Topology file', type=str,
                        default="./topologies/conv_nets/test.csv",
                        help="Path to the topology file"
                        )
    parser.add_argument('-c', metavar='Config file', type=str,
                        default="./configs/scale.cfg",
                        help="Path to the config file"
                        )
    parser.add_argument('-p', metavar='log dir', type=str,
                        default="./test_runs",
                        help="Path to log dir"
                        )
    parser.add_argument('-i', metavar='input type', type=str,
                        default="conv",
                        help="Type of input topology, gemm: MNK, conv: conv"
                        )

    args = parser.parse_args()
    topology = args.t
    config = args.c
    logpath = args.p
    inp_type = args.i

    gemm_input = False
    if inp_type == 'gemm':
        gemm_input = True

    # s = scalesim(save_disk_space=True, verbose=True,
    #              config=config,
    #              topology=topology,
    #              input_type_gemm=gemm_input
    #              )
    # s.run_scale(top_path=logpath)
    top_path=logpath
    s_sim = sim()
    
    s_config = scale_config()
    s_config.read_conf_file(config)
    s_topology = topologies()
    s_topology.load_arrays_conv(topofile=topology)

    s_sim.set_params(
            config_obj=s_config,
            topo_obj=s_topology,
            top_path=top_path,
            verbosity=True,
            save_trace=True
    )

    print(top_path)

    s_sim.run()


