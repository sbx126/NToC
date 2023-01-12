
function input_condprobs_kernel(nodes, input_node_ids, heap, mars, flows, condprobs, num_ex_threads::Int32, node_work::Int32)
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_ids))

    @inbounds if ex_id <= size(mars, 1)
        for node_id = node_start : node_end
            orig_node_id::UInt32 = input_node_ids[node_id]
            node_flow::Float32 = flows[ex_id, orig_node_id]
            inputnode = nodes[orig_node_id]::PCs.BitsInput
            variable = inputnode.variable
            distrib = PCs.dist(inputnode)
            if distrib.value # pos literal
                condprobs[ex_id, variable, 1] = node_flow
            else # neg literal
                condprobs[ex_id, variable, 2] = node_flow
            end
        end
    end
    nothing
end

function input_condprobs(bpc, mars, flows, condprobs)
    condprobs .= zero(Float32)

    num_input_nodes = length(bpc.input_node_ids)
    num_examples = size(mars, 1)

    dummy_args = (bpc.nodes, bpc.input_node_ids, bpc.heap, mars, flows, condprobs, Int32(1), Int32(1))
    kernel = @cuda name="input_condprobs" launch=false input_condprobs_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        PCs.balance_threads(num_input_nodes, num_examples, config; mine = 2, maxe = 32)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, mars, flows, condprobs, Int32(num_example_threads), Int32(node_work))
    kernel(args...; threads, blocks)
    nothing
end

function pc_condprobs(pc::ProbCircuit, data::Matrix{Float32}, target_vars; batch_size = 128, 
                      mars_mem = nothing, flows_mem = nothing, probs_mem = nothing, edge_aggr_mem = nothing)
    bpc = PCs.CuBitsProbCircuit(pc)
    pc_condprobs(bpc, data, target_vars; batch_size, mars_mem, flows_mem, probs_mem, edge_aggr_mem)
end
function pc_condprobs(bpc::PCs.CuBitsProbCircuit, data::Matrix{Float32}, target_vars; batch_size = 128, 
                      mars_mem = nothing, flows_mem = nothing, probs_mem = nothing, edge_aggr_mem = nothing)
    # prepare bitcircuit
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    # prepare data
    num_examples = size(data, 1)
    num_features = size(data, 2)
    new_data = Matrix{Union{Float32,Missing}}(undef, num_examples, num_features)
    new_data .= data
    for v in target_vars
        new_data[:, v] .= missing
    end
    data = cu(new_data)

    # prepare memory
    mars = PCs.prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = PCs.prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    probs = PCs.prep_memory(probs_mem, (batch_size, num_features, 2), (false, true, true))
    edge_aggr = PCs.prep_memory(edge_aggr_mem, (num_edges,))
    all_probs = CUDA.zeros(Float32, num_examples, num_features, 2)

    for batch_start = 1 : batch_size : num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        PCs.probs_flows_circuit(flows, mars, edge_aggr, bpc, data, batch; mine = 2, maxe = 32)
        input_condprobs(bpc, mars, flows, probs)

        all_probs[batch_start:batch_end, :, :] .= @view probs[1:num_batch_examples, :, :]
    end

    all_probs = Array(all_probs)
    all_probs = @views all_probs[:,:,1] ./ (all_probs[:,:,1] .+ all_probs[:,:,2])
    PCs.cleanup_memory((mars, mars_mem), (flows, flows_mem), (probs, probs_mem), (edge_aggr, edge_aggr_mem))

    all_probs[:,target_vars]
end
