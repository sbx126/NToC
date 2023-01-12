
function pc_values(pc::ProbCircuit, data::Matrix{Float32}; batch_size = 128, mars_mem = nothing)
    bpc = PCs.CuBitsProbCircuit(pc)
    pc_values(bpc, data; batch_size, mars_mem)
end
function pc_values(bpc::PCs.CuBitsProbCircuit, data::Matrix{Float32}; batch_size = 128, mars_mem = nothing)
    # prepare bitcircuit
    num_nodes = length(bpc.nodes)

    # prepare data
    data = cu(data)
    num_examples = size(data, 1)

    # prepare memory
    mars = PCs.prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    vals = CUDA.zeros(Float32, num_examples)
    
    for batch_start = 1 : batch_size : num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start : batch_end
        num_batch_examples = length(batch)

        PCs.eval_circuit(mars, bpc, data, batch; mine = 2, maxe = 32)
        
        vals[batch_start:batch_end] .= @view mars[1:num_batch_examples, end]
    end

    vals = Array(vals)
    PCs.cleanup_memory(mars, mars_mem)

    vals
end
