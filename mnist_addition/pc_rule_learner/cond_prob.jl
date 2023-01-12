# -*- coding: utf-8 -*-


"Conditional probability given a LogicCircuit and soft inputs"
function lc_condprob(lc::LogicCircuit, data::Matrix{Float}, target_vars::Vector{<:Integer};
                     w_cache = nothing,
                     v_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}(),
                     f_cache::Dict{LogicCircuit,Vector{Float}} = Dict{LogicCircuit,Vector{Float}}()) where Float <: AbstractFloat
    n_examples = size(data, 1)

    # Clip data
    data = clamp!(data, 1e-6, 1.0 - 1e-6)

    # Compute values
    f_v_con(_)::Vector{Float} = error("Does not support constant nodes.")
    f_v_lit(n)::Vector{Float} = begin
        v = lit2var(n.literal)
        if v in target_vars 
            # Need to marginalize out `v`
            zeros(Float, n_examples) # Assign log-probability log(1.0) everywhere
        else
            if dist(n).value
                log.(data[:, v])
            else
                log.(1.0 .- data[:, v])
            end
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
                for (i, c) in enumerate(children(n))
                    edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ ws[i]
                    if haskey(f_cache, c)
                        f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                    else
                        f_cache[c] = deepcopy(edge_flow)
                    end
                end
            else
                for c in children(n)
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
            for c in children(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end

    target_flows::Array{Float,3} = zeros(Float, n_examples, length(target_vars), 2)
    foreach(lc) do n
        if isliteralgate(n)
            v = lit2var(n.literal)
            if v in target_vars
                v_idx = findall(target_vars .== v)[1]
                if n.literal > zero(Lit)
                    @inbounds @views target_flows[:,v_idx,1] .= f_cache[n]
                else
                    @inbounds @views target_flows[:,v_idx,2] .= f_cache[n]
                end
            end
        end
    end
    cond_probs::Matrix{Float} = zeros(Float, n_examples, length(target_vars))
    for v = 1 : length(target_vars)
        @inbounds @views cond_probs[:,v] .= exp.(target_flows[:,v,1] .- logaddexp.(target_flows[:,v,1], target_flows[:,v,2]))
    end

    cond_probs
end


"Conditional probability given a ProbCircuit and soft inputs"
function pc_condprob(pc::ProbCircuit, data::Union{Matrix{Float},Matrix{Union{Float,Missing}}}, target_vars::Vector{<:Integer};
                     v_cache::Dict{ProbCircuit,Vector{Float}} = Dict{ProbCircuit,Vector{Float}}(),
                     f_cache::Dict{ProbCircuit,Vector{Float}} = Dict{ProbCircuit,Vector{Float}}(), debug::Bool = false) where Float <: AbstractFloat
    n_examples = size(data, 1)

    # Clip data
    data = clamp!(data, 1e-6, 1.0 - 1e-6)

    # Compute values
    f_v_lit(n)::Vector{Float} = begin
        v = randvar(n)
        if v in target_vars 
            # Need to marginalize out `v`
            zeros(Float, n_examples) # Assign log-probability log(1.0) everywhere
        else
            if dist(n).value
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
    foldup_aggregate(pc, f_v_lit, f_v_a, f_v_o, Vector{Float}, v_cache)

    # Compute flows
    f_cache[pc] = v_cache[pc]
    folddown_each(pc) do n
        if issum(n)
            for (i, c) in enumerate(children(n))
                edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ n.params[i]
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        elseif ismul(n)
            edge_flow = f_cache[n]
            for c in children(n)
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end

    target_flows::Array{Float,3} = zeros(Float, n_examples, length(target_vars), 2)
    foreach(pc) do n
        if isinput(n)
            v = randvar(n)
            if v in target_vars
                v_idx = findall(target_vars .== v)[1]
                if dist(n).value
                    @inbounds @views target_flows[:,v_idx,1] .= f_cache[n]
                else
                    @inbounds @views target_flows[:,v_idx,2] .= f_cache[n]
                end
            end
        end
    end
    cond_probs::Matrix{Float} = zeros(Float, n_examples, length(target_vars))
    for v = 1 : length(target_vars)
        @inbounds @views cond_probs[:,v] .= exp.(target_flows[:,v,1] .- logaddexp.(target_flows[:,v,1], target_flows[:,v,2]))
    end

    if debug
        cond_probs, v_cache, f_cache
    else
        cond_probs
    end
end
