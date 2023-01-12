
function input_grads_kernel(nodes, input_node_ids, heap, mars, flows, grads, eps::Float32, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(mars, 1)
        for node_id = node_start : node_end
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_value::Float32 = exp(mars[ex_id, orig_node_id])
            node_flow::Float32 = flows[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::PCs.BitsInput
            variable = inputnode.variable
            distrib = PCs.dist(inputnode)
            if distrib.value # pos literal
                g = node_flow / (node_value + eps)
                CUDA.@atomic grads[ex_id, variable] += g
            else # neg literal
                g = node_flow / (node_value + eps)
                CUDA.@atomic grads[ex_id, variable] -= g
            end
        end
    end
    nothing
end

function input_grads(bpc, mars, flows, grads; eps = 1e-3)
    grads .= zero(Float32)

    num_input_nodes = length(bpc.input_node_ids)
    num_examples = size(mars, 1)

    dummy_args = (bpc.nodes, bpc.input_node_ids, bpc.heap, mars, flows, grads, Float32(eps), Int32(1), Int32(1))
    kernel = @cuda name="input_grads" launch=false input_grads_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        PCs.balance_threads(num_input_nodes, num_examples, config; mine = 2, maxe = 32)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, mars, flows, grads, Float32(eps), Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

function pc_gradients(pc::ProbCircuit, data::Matrix{Float32}; batch_size = 128, log_grad = false, marginalized_vars = [],
                      mars_mem = nothing, flows_mem = nothing, grads_mem = nothing, edge_aggr_mem = nothing)
    bpc = PCs.CuBitsProbCircuit(pc)
    pc_gradients(bpc, data; batch_size, log_grad, marginalized_vars, mars_mem, flows_mem, grads_mem, edge_aggr_mem)
end
function pc_gradients(bpc::PCs.CuBitsProbCircuit, data::Matrix{Float32}; batch_size = 128, log_grad = false, marginalized_vars = [],
                      mars_mem = nothing, flows_mem = nothing, grads_mem = nothing, edge_aggr_mem = nothing, eps = 1e-3)
    # prepare bitcircuit
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    # prepare data
    num_examples = size(data, 1)
    num_features = size(data, 2)
    if length(marginalized_vars) > 0
        new_data = Matrix{Union{Float32,Missing}}(undef, num_examples, num_features)
        new_data .= data
        for v in marginalized_vars
            new_data[:, v] .= missing
        end
        data = new_data
    end
    data = cu(data)
    
    mars = PCs.prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = PCs.prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    grads = PCs.prep_memory(grads_mem, (batch_size, num_features), (false, true))
    edge_aggr = PCs.prep_memory(edge_aggr_mem, (num_edges,))
    all_grads = CUDA.zeros(Float32, num_examples, num_features)

    for batch_start = 1 : batch_size : num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        PCs.probs_flows_circuit(flows, mars, edge_aggr, bpc, data, batch; mine = 2, maxe = 32)
        input_grads(bpc, mars, flows, grads; eps)
        if log_grad
            grads[:,:] ./= exp.(mars[:,end:end])
        end

        all_grads[batch_start:batch_end, :] .= @view grads[1:num_batch_examples, :]
    end

    all_grads = Array(all_grads)
    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (grads, grads_mem), (edge_aggr, edge_aggr_mem))
    all_grads
end
