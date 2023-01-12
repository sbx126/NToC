
function full_batch_em(pc::ProbCircuit, data::Matrix{Float32}; step_size = 0.01, batch_size = 128, pseudocount = 0.1,
                       mars_mem = nothing, flows_mem = nothing, edge_aggr_mem = nothing, node_aggr_mem = nothing)
    bpc = PCs.CuBitsProbCircuit(pc)
    full_batch_em(bpc, data; step_size, batch_size, pseudocount, mars_mem, flows_mem, edge_aggr_mem, node_aggr_mem)

    # update parameters to ProbCircuit
    PCs.update_parameters(bpc)
    nothing
end
function full_batch_em(bpc::PCs.CuBitsProbCircuit, data::Matrix{Float32}; step_size = 0.01, batch_size = 128, pseudocount = 0.1,
                       mars_mem = nothing, flows_mem = nothing, edge_aggr_mem = nothing, node_aggr_mem = nothing)
    # prepare bitcircuit
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    # prepare data
    data = cu(data)
    num_examples = size(data, 1)

    # prepare memory
    mars = PCs.prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = PCs.prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    edge_aggr = PCs.prep_memory(edge_aggr_mem, (num_edges,))
    node_aggr = PCs.prep_memory(node_aggr_mem, (num_nodes,))

    edge_aggr .= zero(Float32)
    PCs.clear_input_node_mem(bpc; rate = 0)

    for batch_start = 1 : batch_size : num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        PCs.probs_flows_circuit(flows, mars, edge_aggr, bpc, data, batch; mine = 2, maxe = 32)
    end

    # update parameters to bpc
    PCs.add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
    PCs.aggr_node_flows(node_aggr, bpc, edge_aggr)
    PCs.update_params(bpc, node_aggr, edge_aggr; inertia = 1.0 - step_size)
    PCs.update_input_node_params(bpc; pseudocount, inertia = 1.0 - step_size)

    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (edge_aggr, edge_aggr_mem), (node_aggr, node_aggr_mem))

    nothing
end
