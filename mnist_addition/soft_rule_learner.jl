# -*- coding: utf-8 -*-
include("./pc_rule_learner/rule_learner.jl")


function rule_learning(lc::LogicCircuit, data::Matrix{Float}; num_targets::Integer, maxiters::Integer, 
        user_threshold::AbstractFloat = 0.002, user_discard::AbstractFloat = 0.005, user_ll::AbstractFloat = 2.0, 
        verbose::Bool = false) where Float <: AbstractFloat
    discard_frac = user_discard
    ## Primitives ##
    ## 1. Get candidate edge-variable pair for SPLIT
    get_split_candidate(lc::LogicCircuit, data::Matrix{Float}; discard_frac::AbstractFloat = user_discard) = begin
        data_idxs = sortperm(lc_values(lc, data)[lc])[Int(ceil(size(data, 1) * discard_frac)):end]
        data = deepcopy(data[data_idxs, :])

        f_cache = lc_flows(lc, data)
        lit_scope_cache = lc_lit_scope(lc)
        num_vars = num_variables(lc)
        best_or, best_and = nothing, nothing
        best_var, best_score = 0, 0.0
        for (n, _) in f_cache
            if !is⋁gate(n) # Only or nodes are considered as candidates
                continue
            end
            for m in children(n)
                if !is⋀gate(m) # Only (or, and) pairs are considered as candidates
                    continue
                end
                m_flows = f_cache[m]
                sum_flow = logsumexp(m_flows)
                mean_flow = sum_flow - log(length(m_flows))
                @inbounds @views var_activs = logsumexp(reshape(m_flows, :, 1) .+ log.(data); dims = 1)
                frac = exp.(var_activs .- sum_flow)
                @inbounds @views entropy = .-(frac .* log.(max.(frac, 0.0)) .+ (1 .- frac) .* log.(max.(1 .- frac, 0.0)))
                @inbounds @views entropy[isnan.(entropy)] .= 0.0
                for v = 1 : num_vars
                    if !(v in lit_scope_cache[m][1]) || !(v in lit_scope_cache[m][2])
                        entropy[v] = 0.0
                    end
                end
                scores = map(1:num_vars) do v
                    if !(v in lit_scope_cache[m][1]) || !(v in lit_scope_cache[m][2])
                        0.0
                    else
                        exp(mean_flow) * entropy[v]
                    end
                end
                score = maximum(scores)
                var = argmax(scores)
                if score > best_score
                    best_score = score
                    best_or = n
                    best_and = m
                    best_var = Var(var)
                end
            end
        end
    
        if best_score < 1e-4
            nothing, nothing, nothing
        else
            best_or, best_and, best_var
        end
    end

    ## 2. Compute semantic loss
    aveg_ll(lc::LogicCircuit, data::Matrix{Float}; discard_frac::Float = user_discard) = begin
        lls = sort(lc_values(lc, data)[lc])
        mean(lls[Int(ceil(length(lls) * discard_frac)):end])
    end

    ## 3. Prune the LC
    prune_lc(lc::LogicCircuit, data::Matrix{Float}, data_mar::Matrix{Union{Float,Missing}}; v_tol::Float, 
        p_threshold::AbstractFloat) = begin
        num_vars = size(data, 2)
        f_cache = lc_flows(lc, data)
        f_mar_cache = lc_flows(lc, data_mar)

        f_con(_)::Union{LogicCircuit,Nothing} = error("Does not support constant nodes.")
        f_lit(n)::Union{LogicCircuit,Nothing} = n
        f_a(n, cns)::Union{LogicCircuit,Nothing} = begin
            if any(cns .=== nothing)
                nothing
            else
                n_flows = deepcopy(f_cache[n])
                criterion = (n_flows .> v_tol)
                if mean(criterion) < p_threshold
                    # Condition 1: if less than `p_threshold` of the training samples have flow greater than `v_tol`,
                    #              this node can potentially be pruned
                    mflow = mean(n_flows)
                    mmarflow = mean(f_mar_cache[n])
                    if mflow == -Inf || mmarflow > mflow # If pruning this node leads to improved `Objective``
                        n # nothing (temporarily disable this)
                    else
                        n
                    end
                else
                    # Condition 2: 
                    lits_to_conjoin = Vector{Lit}()
                    for v = Var(1) : Var(num_vars)
                        flag = true

                        # Expected decrease in semantic loss if conjoin with (v)
                        if flag && mean(data[criterion, v] .< 0.9) < p_threshold
                            push!(lits_to_conjoin, var2lit(v))
                            flag = false
                        end

                        # Expected decrease in semantic loss if conjoin with (-v)
                        if flag && mean(data[criterion, v] .> 0.1) < p_threshold
                            push!(lits_to_conjoin, - var2lit(v))
                            flag = false
                        end
                    end

                    # Conjoin with the selected variables
                    new_n = conjoin(cns...)
                    for l in lits_to_conjoin
                        conj_n = conjoin(new_n, l)
                        if !has_const_node(conj_n)
                            new_n = conj_n
                        end
                    end
                    new_n
                end
            end
        end
        f_o(n, cns)::Union{LogicCircuit,Nothing} = begin
            if all(cns .=== nothing)
                nothing
            else
                new_cns = filter(n -> n !== nothing, cns)
                disjoin(new_cns...)
            end
        end
        foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Union{LogicCircuit,Nothing})
    end

    ## Prepair dataset ##
    data_mar = Matrix{Union{Float,Missing}}(undef, size(data, 1), size(data, 2))
    @inbounds @views data_mar .= data
    @inbounds @views data_mar[:,size(data, 2)-num_targets+1:end] .= missing

    ## Iteratively improve the logical sentence ##
    iter_idx = 1
    while iter_idx < maxiters
        ## Split the Logic Circuit ##
        # Compute the best candidate
        or, and, var = get_split_candidate(lc, data)
        # Perform split
        if or !== nothing
            lc, or = split_lc(lc, (or, and), var; depth = 1)
        else
            if verbose
                open("rule.log", "a+") do io
                    write(io, "> Split candidate not found\n")
                end
            end
        end

        ## Prune the Logic Circuit ##
        v_tol = aveg_ll(lc, data) - log(user_ll)
        p_threshold = user_threshold
        lc = prune_lc(lc, data, data_mar; v_tol, p_threshold)

        if verbose
            println("Iteration #$(iter_idx)")
            println("> Aveg pos sample LL: $(aveg_ll(lc, data))")
            println("> Model count: $(model_count(lc))")
            println("> # edges: $(num_edges(lc)); # nodes: $(num_nodes(lc))")
            println(lc)
        end
        iter_idx += 1
    end

    lc
end

function check_precision_base(lc::LogicCircuit, base::Integer)
    get_data(c1, c2, c3, c_in, c_out, base) = begin
        d = zeros(Bool, 3 * base + 4)
        @inbounds d[c1] = true
        @inbounds d[c2 + base] = true
        @inbounds d[c3 + 2 * base + 4] = true
        @inbounds d[c_in + 2 * base] = true
        @inbounds d[c_out + 2 * base + 2] = true
        d
    end

    count = 0
    for c1 = 1 : base
        for c2 = 1 : base
            for c_in = 1 : 2
                c3 = (c1-1 + c2-1 + c_in-1) % base + 1
                c_out = ((c1-1 + c2-1 + c_in-1) ÷ base) +1
                
                if lc(get_data(c1, c2, c3, c_in, c_out, base)...)
                    count += 1
                    #println(c1-1,"+",c2-1,"+",c_in-1,"=",c_out-1,c3-1)
                end
            end
        end
    end

    count 
end
