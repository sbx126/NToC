# -*- coding: utf-8 -*-
using StatsFuns: logaddexp, logsubexp
using DataFrames: DataFrame


function lc_gradient(lc::LogicCircuit, data::Matrix{Float}; log_grad = false, w_cache = nothing,
                      v_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}(),
                      f_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}()) where Float <: AbstractFloat
    n_examples = size(data, 1)

    # Clip data
    data = clamp!(data, 1e-6, 1.0 - 1e-6)

    # Compute values
    f_v_con(_)::Vector{Float} = error("Does not support constant nodes.")
    f_v_lit(n)::Vector{Float} = begin
        v = lit2var(n.literal)
        if n.literal > zero(Lit)
            log.(data[:, v])
        else
            log.(1.0 .- data[:, v])
        end
    end
    f_v_a(_, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_v_o(n, cn)::Vector{Float} = begin
        if w_cache !== nothing && haskey(w_cache, n)
            ws = w_cache[n]
            map(1:n_examples) do idx
                logsumexp([v[idx] + ws[i] for (i, v) in enumerate(cn)])
            end
        else
            map(1:n_examples) do idx
                logsumexp([v[idx] for v in cn])
            end
        end
    end
    foldup_aggregate(lc, f_v_con, f_v_lit, f_v_a, f_v_o, Vector{Float}, v_cache)

    # Compute flows
    f_cache[lc] = v_cache[lc]
    folddown_each(lc) do n
        if is⋁gate(n)
            if w_cache !== nothing && haskey(w_cache, n)
                ws = w_cache[n]
                for (i, c) in enumerate(inputs(n))
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ ws[i]
                    if haskey(f_cache, c)
                        f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                    else
                        f_cache[c] = deepcopy(edge_flow)
                    end
                end
            else
                for c in inputs(n)
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n]
                    if haskey(f_cache, c)
                        f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                    else
                        f_cache[c] = deepcopy(edge_flow)
                    end
                end
            end
        elseif is⋀gate(n)
            edge_flow = f_cache[n]
            for c in inputs(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end
    
    # Compute gradients
    var_grads = zeros(Float, n_examples, size(data, 2), 2)
    foreach(lc) do n
        if isliteralgate(n)
            l = n.literal
            v = lit2var(l)
            if l > zero(Lit)
                # Positive literal
                @inbounds @views var_grads[:,v,1] .= f_cache[n] .- v_cache[n]
            else
                # Negative literal
                @inbounds @views var_grads[:,v,2] .= f_cache[n] .- v_cache[n]
            end
        end
    end

    if log_grad
        var_grads = ifelse.(var_grads[:,:,1] .>= var_grads[:,:,2], 
                            exp.(logsubexp.(var_grads[:,:,1], var_grads[:,:,2]) .- reshape(v_cache[lc], :, 1)),
                            .- exp.(logsubexp.(var_grads[:,:,2], var_grads[:,:,1]) .- reshape(v_cache[lc], :, 1)))
    else
        var_grads = ifelse.(var_grads[:,:,1] .>= var_grads[:,:,2], 
                            exp.(logsubexp.(var_grads[:,:,1], var_grads[:,:,2])),
                            .- exp.(logsubexp.(var_grads[:,:,2], var_grads[:,:,1])))
    end

    var_grads, v_cache[lc]
end


function lc_gradients_cat(lc::LogicCircuit, data::Matrix{Float}, cat_lits::Vector{T}; log_grad = false, w_cache = nothing,
                          v_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}(),
                          f_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}()) where {Float <: AbstractFloat, T <: LogicCircuit}
    n_examples = size(data, 1)

    # Clip data
    data = clamp!(data, 1e-6, 1.0 - 1e-6)

    # Compute values
    f_v_con(_)::Vector{Float} = error("Does not support constant nodes.")
    f_v_lit(n)::Vector{Float} = begin
        v = lit2var(n.literal)
        if n.literal > zero(Lit)
            log.(data[:, v])
        else
            log.(1.0 .- data[:, v])
        end
    end
    f_v_a(_, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_v_o(n, cn)::Vector{Float} = begin
        if w_cache !== nothing && haskey(w_cache, n)
            ws = w_cache[n]
            map(1:n_examples) do idx
                logsumexp([v[idx] + ws[i] for (i, v) in enumerate(cn)])
            end
        else
            map(1:n_examples) do idx
                logsumexp([v[idx] for v in cn])
            end
        end
    end
    foldup_aggregate(lc, f_v_con, f_v_lit, f_v_a, f_v_o, Vector{Float}, v_cache)

    # Compute flows
    f_cache[lc] = v_cache[lc]
    folddown_each(lc) do n
        if is⋁gate(n)
            if w_cache !== nothing && haskey(w_cache, n)
                ws = w_cache[n]
                for (i, c) in enumerate(inputs(n))
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ ws[i]
                    if haskey(f_cache, c)
                        f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                    else
                        f_cache[c] = deepcopy(edge_flow)
                    end
                end
            else
                for c in inputs(n)
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n]
                    if haskey(f_cache, c)
                        f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                    else
                        f_cache[c] = deepcopy(edge_flow)
                    end
                end
            end
        elseif is⋀gate(n)
            edge_flow = f_cache[n]
            for c in inputs(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end

    # Compute gradients
    var_grads::Matrix{Float} = zeros(Float, n_examples, length(cat_lits))
    @inbounds @views var_grads[:,:] .= typemin(Float)
    foreach(lc) do n
        if n in cat_lits
            lit_idx = 0
            for i = 1 : length(cat_lits)
                if n === cat_lits[i]
                    lit_idx = i
                    break
                end
            end

            @inbounds @views var_grads[:,lit_idx] .= logaddexp.(var_grads[:,lit_idx], f_cache[n] .- v_cache[n])
        end
    end

    if log_grad
        var_grads[:,:] .-= reshape(v_cache[lc], :, 1)
    end

    var_grads, v_cache[lc]
end


function pc_gradient(pc::ProbCircuit, data::Union{Matrix{Float},Matrix{Union{Float,Missing}}}; log_grad::Bool = false,
                      marginalized_vars::Vector{Int} = Vector{Int}(), no_prob::Bool = false,
                      v_cache::Dict{ProbCircuit,Vector{Float}} = Dict{ProbCircuit,Vector{Float}}(),
                      f_cache::Dict{ProbCircuit,Vector{Float}} = Dict{ProbCircuit,Vector{Float}}()) where Float <: AbstractFloat
    n_examples = size(data, 1)

    # Clip data
    data = map(v -> ifelse(ismissing(v), missing, clamp(v, 1e-6, 1.0 - 1e-6)), data)

    # Compute values
    f_v_con(_)::Vector{Float} = error("Does not support constant nodes.")
    f_v_lit(n)::Vector{Float} = begin
        #v = lit2var(n.randvars)
        v = first(n.randvars)
        #first(n.randvars)
        #randvar()
        if v in marginalized_vars
            zeros(Float, n_examples)
        else
            if first(n.randvars) > zero(Lit)
                log.(coalesce.(data[:, v], 1.0))
            else
                log.(1.0 .- coalesce.(data[:, v], 0.0))
            end
        end
    end
    f_v_a(_, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_v_o(n, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            if no_prob
                logsumexp([v[idx] for (i, v) in enumerate(cn)])
            else
                logsumexp([v[idx] + n.params[i] for (i, v) in enumerate(cn)])
            end
        end
    end
    foldup_aggregate(pc, f_v_lit, f_v_a, f_v_o, Vector{Float}, v_cache)

    # Compute flows
    f_cache[pc] = v_cache[pc]
    f_edge_cache = Dict{Tuple{ProbCircuit,ProbCircuit},Vector{Float64}}()
    folddown_each(pc) do n
        if issum(n)
            for (i, c) in enumerate(inputs(n))
                if no_prob
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n]
                else
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ n.params[i]
                end
                f_edge_cache[(n, c)] = deepcopy(edge_flow)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        elseif ismul(n)
            edge_flow = f_cache[n]
            for c in inputs(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end

    # Compute gradients
    var_grads = zeros(Float, n_examples, size(data, 2), 2)
    node_flows = Dict{ProbCircuit,Vector{Float64}}()
    foreach(pc) do n
        if isinput(n)
            l = first(n.randvars)
            #v = lit2var(l)
            v = l
            if l > zero(Lit)
                # Positive literal
                @inbounds @views var_grads[:,v,1] .= f_cache[n] .- v_cache[n]
            else
                # Negative literal
                @inbounds @views var_grads[:,v,2] .= f_cache[n] .- v_cache[n]
            end
        elseif issum(n)
            n_flows = Vector{Float64}(undef, num_children(n))
            for (i, c) in enumerate(inputs(n))
                n_flows[i] = logsumexp(f_edge_cache[(n, c)]) - log(n_examples)
            end
            n_flows .-= logsumexp(n_flows)
            node_flows[n] = n_flows
        end
    end

    if log_grad
        var_grads = ifelse.(var_grads[:,:,1] .>= var_grads[:,:,2], 
                            exp.(logsubexp.(var_grads[:,:,1], var_grads[:,:,2]) .- reshape(v_cache[pc], :, 1)),
                            .- exp.(logsubexp.(var_grads[:,:,2], var_grads[:,:,1]) .- reshape(v_cache[pc], :, 1)))
    else
        var_grads = ifelse.(var_grads[:,:,1] .>= var_grads[:,:,2], 
                            exp.(logsubexp.(var_grads[:,:,1], var_grads[:,:,2])),
                            .- exp.(logsubexp.(var_grads[:,:,2], var_grads[:,:,1])))
    end

    var_grads, v_cache[pc], node_flows
end


function pc_input_flows(pc::ProbCircuit, data::Union{Matrix{Float},Matrix{Union{Float,Missing}}};
                        marginalized_vars::Vector{Int} = Vector{Int}(),
                        v_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}(),
                        f_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}()) where Float <: AbstractFloat
    n_examples = size(data, 1)

    # Clip data
    data = map(v -> ifelse(ismissing(v), missing, clamp(v, 1e-6, 1.0 - 1e-6)), data)

    # Compute values
    f_v_con(_)::Vector{Float} = error("Does not support constant nodes.")
    f_v_lit(n)::Vector{Float} = begin
        v = lit2var(n.literal)
        if v in marginalized_vars
            zeros(Float, n_examples)
        else
            if n.literal > zero(Lit)
                log.(coalesce.(data[:, v], 1.0))
            else
                log.(1.0 .- coalesce.(data[:, v], 0.0))
            end
        end
    end
    f_v_a(_, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_v_o(n, cn)::Vector{Float} = begin
        map(1:n_examples) do idx
            logsumexp([v[idx] + n.params[i] for (i, v) in enumerate(cn)])
        end
    end
    foldup_aggregate(pc, f_v_con, f_v_lit, f_v_a, f_v_o, Vector{Float}, v_cache)

    # Compute flows
    f_cache[pc] = v_cache[pc]
    f_edge_cache = Dict{Tuple{ProbCircuit,ProbCircuit},Vector{Float64}}()
    folddown_each(pc) do n
        if is⋁gate(n)
            for (i, c) in enumerate(inputs(n))
                edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ n.params[i]
                f_edge_cache[(n, c)] = deepcopy(edge_flow)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        elseif is⋀gate(n)
            edge_flow = f_cache[n]
            for c in inputs(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end
    
    # Compute input flows
    var_flows = zeros(Float, n_examples, size(data, 2), 2)
    node_flows = Dict{ProbCircuit,Vector{Float64}}()
    foreach(pc) do n
        if isliteralgate(n)
            l = n.literal
            v = lit2var(l)
            if l > zero(Lit)
                # Positive literal
                @inbounds @views var_flows[:,v,1] .= f_cache[n]
            else
                # Negative literal
                @inbounds @views var_flows[:,v,2] .= f_cache[n]
            end
        elseif is⋁gate(n)
            n_flows = Vector{Float64}(undef, num_children(n))
            for (i, c) in enumerate(inputs(n))
                n_flows[i] = logsumexp(f_edge_cache[(n, c)]) - log(n_examples)
            end
            n_flows .-= logsumexp(n_flows)
            node_flows[n] = n_flows
        end
    end
    @inbounds @views var_flows = exp.(var_flows[:,:,1]) ./ (exp.(var_flows[:,:,1]) .+ exp.(var_flows[:,:,2]))
    
    var_flows, v_cache[pc], node_flows
end


function accum_pc_flows!(node_flows_target::Dict, node_flows::Dict)
    for n in keys(node_flows)
        if haskey(node_flows_target, n)
            node_flows_target[n] = logaddexp.(node_flows_target[n], node_flows[n])
        else
            node_flows_target[n] = deepcopy(node_flows[n])
        end
        node_flows_target[n] .-= logsumexp(node_flows_target[n])
    end
    node_flows_target
end


function pc_em_update!(pc::ProbCircuit, node_flows::Dict; step_size::Float) where Float <: Float64
    foreach(pc) do n
        if is⋁gate(n)
            n.params = logaddexp.(n.params .+ log(1.0 - step_size), node_flows[n] .+ log(step_size))
        end
    end
end
function pc_em_update_with_inc!(pc::ProbCircuit, node_flows::Dict; step_size::Float, update_nodes::Vector) where Float <: Float64
    foreach(pc) do n
        if is⋁gate(n) && (n in update_nodes)
            n.params = logaddexp.(n.params .+ log(1.0 - step_size), node_flows[n] .+ log(step_size))
        end
    end
end
function pc_em_update_with_exc!(pc::ProbCircuit, node_flows::Dict; step_size::Float, exclude_nodes::Vector) where Float <: Float64
    foreach(pc) do n
        if is⋁gate(n) && !(n in exclude_nodes)
            n.params = logaddexp.(n.params .+ log(1.0 - step_size), node_flows[n] .+ log(step_size))
        end
    end
end

function pc_param_learn!(pc::ProbCircuit, data::Matrix{Float}; batch_size::Integer, 
                         step_size::Float, pseudocount::Float = 0.1) where Float <: Float64
    data = batch(DataFrame(data, :auto), batch_size)
    estimate_parameters_em!(pc, data; pseudocount, exp_update_factor = 1.0 - step_size, update_per_batch = true)
end
