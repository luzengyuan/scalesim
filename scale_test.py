import argparse

#from scalesim.scale_sim import scalesim
#from scale_sim import scalesim

from scale_config import scale_config
#from scalesim.topology_utils import topologies
from topology_utils import topologies
from simulator import simulator as sim
from compute.operand_matrix import operand_matrix
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad as mem_dbsp

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

    #s = scalesim(save_disk_space=True, verbose=True,
    #             config=config,
    #             topology=topology,
    #             input_type_gemm=gemm_input)
    
    

    #s.run_scale(top_path=logpath)
    print("** Config **")
    s_config = scale_config()
    s_config.read_conf_file(config)
    array_rows= s_config.array_rows
    print("array_rows:",array_rows)
    print()

    print("** Topology **")
    s_topology = topologies()
    s_topology.load_arrays_conv(topofile=topology)
    topo_arrays = s_topology.topo_arrays
    print("topo_arrays:",topo_arrays)
    num_layers = s_topology.num_layers
    print("num_layers:",num_layers)

    layer_param_arr = s_topology.get_layer_params(layer_id=0)
    ofmap_dims = s_topology.get_layer_ofmap_dims(layer_id=0)
    ofmap_px_filt = s_topology.get_layer_num_ofmap_px(layer_id=0) / s_topology.get_layer_num_filters(layer_id=0)
    conv_window_size = s_topology.get_layer_window_size(layer_id=0)
    layer_calc_hyper_param_arr = [ofmap_dims[0], ofmap_dims[1], ofmap_px_filt, conv_window_size]
    config_arr = [512, 512, 256, 8, 8]

    print("layer_calc_hyper_param_arr:",layer_calc_hyper_param_arr)
    print()

    print("** operand_matrix **")
    opmat = operand_matrix()


    opmat.set_params(config_obj=s_config,
                          topoutil_obj=s_topology,
                          layer_id=0)
    ##opmat.matrix_set
    #opmat.get_all_operand_matrix()
    
    ifmap_addr_matrix = opmat.ifmap_addr_matrix
    filter_addr_matrix = opmat.filter_addr_matrix
    ofmap_addr_matrix = opmat.ofmap_addr_matrix
    
    print("ifmap_addr_matrix:",ifmap_addr_matrix.shape)
    print("filter_addr_matrix:",filter_addr_matrix.shape)
    print("ofmap_addr_matrix:",ofmap_addr_matrix.shape)
    print()

    print("** memory_system **")
    memory_system = mem_dbsp()

    word_size = 1           # bytes, this can be incorporated in the config file
    active_buf_frac = 0.5   # This can be incorporated in the config as well

    ifmap_buf_size_kb, filter_buf_size_kb, ofmap_buf_size_kb = s_config.get_mem_sizes()
    ifmap_buf_size_bytes = 1024 * ifmap_buf_size_kb
    filter_buf_size_bytes = 1024 * filter_buf_size_kb
    ofmap_buf_size_bytes = 1024 * ofmap_buf_size_kb
    ifmap_backing_bw = 1
    filter_backing_bw = 1
    ofmap_backing_bw = 1
    estimate_bandwidth_mode = False

    memory_system.set_params(word_size=word_size,
                    ifmap_buf_size_bytes=ifmap_buf_size_bytes,
                    filter_buf_size_bytes=filter_buf_size_bytes,
                    ofmap_buf_size_bytes=ofmap_buf_size_bytes,
                    rd_buf_active_frac=active_buf_frac, wr_buf_active_frac=active_buf_frac,
                    ifmap_backing_buf_bw=ifmap_backing_bw,
                    filter_backing_buf_bw=filter_backing_bw,
                    ofmap_backing_buf_bw=ofmap_backing_bw,
                    verbose=True,
                    estimate_bandwidth_mode=estimate_bandwidth_mode)
    
    ifmap_buf_total_size_bytes = memory_system.ifmap_buf.total_size_bytes
    
    print("ifmap_buf_total_size_bytes:",ifmap_buf_total_size_bytes)

