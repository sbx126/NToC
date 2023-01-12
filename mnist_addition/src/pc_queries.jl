# -*- coding: utf-8 -*-


function pc_value(pc::ProbCircuit, data::Union{Matrix{Float64},Matrix{Union{Float64,Missing}}}; 
                   v_cache::Dict{ProbCircuit,Vector{Float64}} = Dict{ProbCircuit,Vector{Float64}}())
    n_examples = size(data, 1)

    f_con(_)::Vector{Float64} = error("Does not support constant nodes.")
    f_lit(n)::Vector{Float64} = begin
        v = lit2var(n.literal)
        if n.literal > zero(Lit)
            log.(coalesce.(data[:, v], 1.0))
        else
            log.(1.0 .- coalesce.(data[:, v], 0.0))
        end
    end
    f_a(_, cn)::Vector{Float64} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_o(n, cn)::Vector{Float64} = begin
        map(1:n_examples) do idx
            logsumexp([v[idx] .+ n.params[i] for (i, v) in enumerate(cn)])
        end
    end
    foldup_aggregate(pc, f_lit, f_a, f_o, Vector{Float64}, v_cache)

    v_cache
end


function pc_flow(pc::LogicCircuit, data::Union{Matrix{Float64},Matrix{Union{Float64,Missing}}};
                  v_cache = nothing, f_cache = Dict{LogicCircuit, Vector{Float64}}())
    if v_cache === nothing
        @assert data !== nothing "Either `data` or `v_cache` has to be provided."
        v_cache = lc_values(pc, data)
    end

    # Assign flow to the root node
    f_cache[pc] = v_cache[pc]

    folddown_each(pc) do n
        if is⋁gate(n)
            for (i, c) in enumerate(children(n))
                edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n] .+ n.log_probs[i]
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        elseif is⋀gate(n)
            for c in children(n)
                edge_flow = deepcopy(f_cache[n])
                if haskey(f_cache, c)
                    f_cache[c] .= logaddexp.(f_cache[c], edge_flow)
                else
                    f_cache[c] = deepcopy(edge_flow)
                end
            end
        end
    end

    f_cache
end
