# -*- coding: utf-8 -*-
using StatsFuns: logaddexp
using Statistics: mean


function lc_values(lc::LogicCircuit, data::Matrix{Bool}; v_cache::Dict{LogicCircuit,Vector{Bool}} = Dict{LogicCircuit,Vector{Bool}}(),
                   pos_lit_missing_treatment = true, neg_lit_missing_treatment = true, applicable_vars::Vector{Var} = Vector{Var}())
    @assert length(applicable_vars) == 0 # For compatibility requirements

    n_examples = size(data, 1)

    f_con(n)::Vector{Bool} = error("Does not support constant nodes.")
    f_lit(n)::Vector{Bool} = begin
        v = lit2var(n.literal)
        if n.literal > zero(Lit)
            deepcopy(data[:, v])
        else
            .!deepcopy(data[:, v])
        end
    end
    f_a(_, cn)::Vector{Bool} = begin
        map(1:n_examples) do idx
            all([v[idx] for v in cn])
        end
    end
    f_o(_, cn)::Vector{Bool} = begin
        map(1:n_examples) do idx
            any([v[idx] for v in cn])
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Vector{Bool}, v_cache)

    v_cache
end
function lc_values(lc::LogicCircuit, data::Matrix{Union{Bool,Missing}}; v_cache::Dict{LogicCircuit,Vector{Bool}} = Dict{LogicCircuit,Vector{Bool}}(),
                   pos_lit_missing_treatment = true, neg_lit_missing_treatment = true, applicable_vars::Vector{Var} = Vector{Var}())
    n_examples = size(data, 1)

    f_con(_)::Vector{Bool} = error("Does not support constant nodes.")
    f_lit(n)::Vector{Bool} = begin
        v = lit2var(n.literal)
        if length(applicable_vars) == 0 || (length(applicable_vars) > 0 && v in applicable_vars)
            if n.literal > zero(Lit)
                deepcopy(coalesce.(data[:, v], pos_lit_missing_treatment))
            else
                .!deepcopy(coalesce.(data[:, v], !neg_lit_missing_treatment))
            end
        else
            if n.literal > zero(Lit)
                deepcopy(coalesce.(data[:, v], true))
            else
                .!deepcopy(coalesce.(data[:, v], false))
            end
        end
    end
    f_a(_, cn)::Vector{Bool} = begin
        map(1:n_examples) do idx
            all([v[idx] for v in cn])
        end
    end
    f_o(_, cn)::Vector{Bool} = begin
        map(1:n_examples) do idx
            any([v[idx] for v in cn])
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Vector{Bool}, v_cache)

    v_cache
end
function lc_values(lc::LogicCircuit, data::Matrix{Float64}; v_cache::Dict{LogicCircuit,Vector{Float64}} = Dict{LogicCircuit,Vector{Float64}}())
    n_examples = size(data, 1)

    f_con(n)::Vector{Float64} = error("Does not support constant nodes.")
    f_lit(n)::Vector{Float64} = begin
        v = lit2var(n.literal)
        if n.literal > zero(Lit)
            log.(data[:, v])
        else
            log.(1.0 .- data[:, v])
        end
    end
    f_a(_, cn)::Vector{Float64} = begin
        map(1:n_examples) do idx
            sum([v[idx] for v in cn])
        end
    end
    f_o(_, cn)::Vector{Float64} = begin
        map(1:n_examples) do idx
            logsumexp([v[idx] for v in cn])
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Vector{Float64}, v_cache)

    v_cache
end
function lc_values(lc::LogicCircuit, data::Matrix{Union{Float64,Missing}}; v_cache::Dict{LogicCircuit,Vector{Float64}} = Dict{LogicCircuit,Vector{Float64}}())
    n_examples = size(data, 1)

    f_con(n)::Vector{Float64} = error("Does not support constant nodes.")
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
    f_o(_, cn)::Vector{Float64} = begin
        map(1:n_examples) do idx
            logsumexp([v[idx] for v in cn])
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Vector{Float64}, v_cache)

    v_cache
end


function lc_flows(lc::LogicCircuit, data::Union{Matrix{Bool},Matrix{Union{Bool,Missing}}}; v_cache = nothing, f_cache = Dict{LogicCircuit, Vector{Bool}}(),
                  pos_lit_missing_treatment = true, neg_lit_missing_treatment = true, applicable_vars::Vector{Var} = Vector{Var}())
    n_examples = size(data, 1)

    if v_cache === nothing
        @assert data !== nothing "Either `data` or `v_cache` has to be provided."
        v_cache = lc_values(lc, data; pos_lit_missing_treatment, neg_lit_missing_treatment, applicable_vars)
    end

    f_con(n, _)::Vector{Bool} = error("Does not support constant nodes.")
    f_lit(n, pns)::Vector{Bool} = begin
        if length(pns) == 0
            zeros(Bool, n_examples)
        else
            pn = map(1:n_examples) do idx
                any([v[idx] for v in pns])
            end
            pn .& v_cache[n]
        end
    end
    f_a(n, pns)::Vector{Bool} = begin
        if length(pns) == 0
            deepcopy(v_cache[n])
        else
            pn = map(1:n_examples) do idx
                any([v[idx] for v in pns])
            end
            pn .& v_cache[n]
        end
    end
    f_o(n, pns)::Vector{Bool} = begin
        if length(pns) == 0
            deepcopy(v_cache[n])
        else
            pn = map(1:n_examples) do idx
                any([v[idx] for v in pns])
            end
            pn .& v_cache[n]
        end
    end
    folddown_aggregate(lc, f_con, f_lit, f_a, f_o, Vector{Bool}, f_cache)

    f_cache
end
function lc_flows(lc::LogicCircuit, data::Matrix{Float64}; v_cache = nothing, f_cache = Dict{LogicCircuit, Vector{Float64}}())
    if v_cache === nothing
        @assert data !== nothing "Either `data` or `v_cache` has to be provided."
        v_cache = lc_values(lc, data)
    end

    # Assign flow to the root node
    f_cache[lc] = v_cache[lc]

    folddown_each(lc) do n
        if is⋁gate(n)
            for c in children(n)
                edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n]
                edge_flow .= ifelse.(isnan.(edge_flow), -Inf, edge_flow)
                if haskey(f_cache, c)
                    f_cache[c] .= deepcopy(logaddexp.(f_cache[c], edge_flow))
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
function lc_flows(lc::LogicCircuit, data::Matrix{Union{Float64,Missing}}; v_cache = nothing, f_cache = Dict{LogicCircuit, Vector{Float64}}())
    if v_cache === nothing
        @assert data !== nothing "Either `data` or `v_cache` has to be provided."
        v_cache = lc_values(lc, data)
    end

    # Assign flow to the root node
    f_cache[lc] = v_cache[lc]

    folddown_each(lc) do n
        if is⋁gate(n)
            for c in children(n)
                edge_flow = v_cache[c] .- v_cache[n] .+ f_cache[n]
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


function lc_zeroflow_nodes(lc, data)
    f_cache = lc_flows(lc, data)
    candidates = Vector{LogicCircuit}()
    for (n, flows) in f_cache
        if sum(flows) == 0
            push!(candidates, n)
        end
    end
    candidates
end

function lc_smallflow_nodes(lc, data; v_tol)
    f_cache = lc_flows(lc, data)
    candidates = Vector{LogicCircuit}()
    mean_flows = Vector{Float64}()
    for (n, flows) in f_cache
        if !is⋀gate(n) # Only prune and nodes to guarantee smoothness
            continue
        end
        mean_flow = mean(flows)
        if mean_flow < v_tol
            push!(candidates, n)
            push!(mean_flows, mean_flow)
        end
    end
    perm_idxs = sortperm(mean_flows)
    total_flows = -Inf
    final_candidates = Vector{LogicCircuit}()
    for i = 1 : length(candidates)
        idx = perm_idxs[i]
        if logaddexp(total_flows, mean_flows[idx]) < v_tol
            push!(final_candidates, candidates[idx])
            total_flows = logaddexp(total_flows, mean_flows[idx])
        end
    end
    final_candidates
end


function has_constant_node(lc)
    foreach(lc) do n
        if n isa PlainConstantNode
            return true
        end
    end
    false
end


function lc_model_count(lc::LogicCircuit; cache::Dict{LogicCircuit,UInt64} = Dict{LogicCircuit,UInt64}())
    f_con(_)::UInt64 = error("Does not support constant node.")
    f_lit(n)::UInt64 = one(UInt64)
    f_a(_, cns)::UInt64 = prod(cns)
    f_o(_, cns)::UInt64 = sum(cns)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, UInt64, cache)

    cache
end


function lc_parents(lc::LogicCircuit; cache::Dict{LogicCircuit,Vector{LogicCircuit}} = Dict{LogicCircuit,Vector{LogicCircuit}}())
    foreach(lc) do n
        if isinner(n)
            for c in children(n)
                if !haskey(cache, c)
                    cache[c] = [n]
                else
                    push!(cache[c], n)
                end
            end
        end
    end
    cache
end


function lc_scope(lc::LogicCircuit; cache::Dict{LogicCircuit,BitSet} = Dict{LogicCircuit,BitSet}())
    f_con(_)::BitSet = error("Does not support constant node.")
    f_lit(n)::BitSet = BitSet(lit2var(n.literal))
    f_a(_, cns)::BitSet = union(cns...)
    f_o(_, cns)::BitSet = intersect(cns...)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, BitSet, cache)
end

function pc_scope(lc::ProbCircuit; cache::Dict{ProbCircuit,BitSet} = Dict{ProbCircuit,BitSet}())
    f_lit(n)::BitSet = BitSet(randvar(n))
    f_a(_, cns)::BitSet = union(cns...)
    f_o(_, cns)::BitSet = intersect(cns...)
    foldup_aggregate(lc, f_lit, f_a, f_o, BitSet, cache)
end

function lc_lit_scope(lc::LogicCircuit; cache::Dict{LogicCircuit,Tuple{BitSet,BitSet}} = Dict{LogicCircuit,Tuple{BitSet,BitSet}}())
    f_con(_)::Tuple{BitSet,BitSet} = error("Does not support constant node.")
    f_lit(n)::Tuple{BitSet,BitSet} = begin
        if n.literal > zero(Lit)
            BitSet(lit2var(n.literal)), BitSet()
        else
            BitSet(), BitSet(lit2var(n.literal))
        end
    end
    f_a(_, cns)::Tuple{BitSet,BitSet} = begin
        union([first(item) for item in cns]...), union([last(item) for item in cns]...)
    end
    f_o(_, cns)::Tuple{BitSet,BitSet} = begin
        union([first(item) for item in cns]...), union([last(item) for item in cns]...)
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Tuple{BitSet,BitSet}, cache)

    cache
end


function has_const_node(lc::LogicCircuit)
    flag = false
    foreach(lc) do n
        if isconstantgate(n)
            flag = true
        end
    end
    flag
end


function lc_get_lits(lc::LogicCircuit)
    num_vars = num_variables(lc)
    lc_lits = Matrix{LogicCircuit}(undef, num_vars, 2)
    foreach(lc) do n
        if isliteralgate(n)
            v = lit2var(n.literal)
            if n.literal > zero(Lit)
                lc_lits[v, 1] = n
            else
                lc_lits[v, 2] = n
            end
        end
    end
    lc_lits
end


function prune_single_model_subcircuit(lc::LogicCircuit, data)
    mc_cache = lc_model_count(lc)
    v_cache = lc_values(lc, data)
    par_cache = lc_parents(lc)

    lc_lits = lc_get_lits(lc)

    f_con(_)::LogicCircuit = error("Does not support constant node.")
    f_lit(n)::LogicCircuit = n
    f_a(n, cns)::LogicCircuit = begin
        if mc_cache[n] > 1 && sum(v_cache[n]) == 1
            flag = true
            for p in par_cache[n]
                if mc_cache[p] == 1
                    flag = false
                    break
                end
            end
            if flag
                scope = lc_scope(n)
                data_idx = argmax(v_cache[n])
                new_cns = Vector{LogicCircuit}()
                for v in scope
                    push!(new_cns, data[data_idx, v] ? lc_lits[v, 1] : lc_lits[v, 2])
                end
                new_n = conjoin(new_cns...)
            else
                new_n = conjoin(cns...)
            end
            new_n
        else
            conjoin(cns...)
        end
    end
    f_o(_, cns)::LogicCircuit = disjoin(cns...)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, LogicCircuit)
end
function prune_single_model_subcircuit_nondet(lc::LogicCircuit, data; num_aux, prune_aux = false)
    num_vars = num_variables(lc) - 2 * num_aux
    f_cache = lc_flows(lc, data)
    lscope_cache = lc_lit_scope(lc)

    f_con(_)::LogicCircuit = error("Does not support constant node.")
    f_lit(n)::LogicCircuit = n
    f_a(n, cns)::LogicCircuit = begin
        n_flows = f_cache[n]
        d = data[n_flows, :]
        new_n = conjoin(cns...)

        lscope_cache = lc_lit_scope(new_n; cache = lscope_cache)
        lits_to_conjoin = Vector{Lit}()
        for v = Var(1) : Var(num_vars) # For each input/target variable (do not consider auxiliary variables)
            if all(d[:, v]) # All true
                if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                    push!(lits_to_conjoin, var2lit(v))
                end
            elseif !any(d[:, v]) # All false
                if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                    push!(lits_to_conjoin, - var2lit(v))
                end
            end
        end
        if prune_aux
            for v = Var(num_vars + 1) : Var(num_vars + 2 * num_aux) # For each auxiliary variable
                if !any(ismissing.(d[:, v]))
                    if all(d[:, v]) # All true
                        if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                            push!(lits_to_conjoin, var2lit(v))
                        end
                    elseif !any(d[:, v]) # All false
                        if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                            push!(lits_to_conjoin, - var2lit(v))
                        end
                    end
                end
            end
        end
        for l in lits_to_conjoin
            new_n = conjoin(new_n, l)
        end
        new_n
    end
    f_o(_, cns)::LogicCircuit = disjoin(cns...)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, LogicCircuit)
end

function prune_single_model_subcircuit_nondet_soft(lc::LogicCircuit, data; v_tol, v_maxtol = v_tol, num_aux = 0, prune_aux = false)
    n_examples = size(data, 1)
    num_vars = num_variables(lc) - 2 * num_aux
    f_cache = lc_flows(lc, data)
    lscope_cache = lc_lit_scope(lc)

    f_con(_)::Union{LogicCircuit,Nothing} = error("Does not support constant node.")
    f_lit(n)::Union{LogicCircuit,Nothing} = n
    f_a(n, cns)::Union{LogicCircuit,Nothing} = begin
        if any(cns .=== nothing)
            nothing
        else
            n_flows = f_cache[n]
            criterion = (n_flows .> v_tol)
            if !any(criterion)
                # All flows are small. In this case should be safe to prune away this node
                nothing
            else
                d = data[criterion, :]
                f = n_flows[criterion, :]
                new_n = conjoin(cns...)

                lscope_cache = lc_lit_scope(new_n; cache = lscope_cache)
                lits_to_conjoin = Vector{Lit}()
                for v = Var(1) : Var(num_vars) # For each input/target variable (do not consider auxiliary variables)
                    if logsumexp(log.(1.0 .- d[:, v]) .+ f) - log(n_examples) < v_maxtol # All close to 1.0
                        if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                            push!(lits_to_conjoin, var2lit(v))

                            v_prune = logsumexp(log.(1.0 .- d[:, v]) .+ f) - log(n_examples)
                            if v_prune > v_maxtol
                                v_maxtol = -Inf
                            else
                                v_maxtol = logsubexp(v_maxtol, v_prune)
                            end
                        end
                    elseif logsumexp(log.(d[:, v]) .+ f) - log(n_examples) < v_maxtol # All close to 0.0
                        if (v in lscope_cache[n][1]) && (v in lscope_cache[n][2])
                            push!(lits_to_conjoin, - var2lit(v))

                            v_prune = logsumexp(log.(d[:, v]) .+ f) - log(n_examples)
                            if v_prune > v_maxtol
                                v_maxtol = -Inf
                            else
                                v_maxtol = logsubexp(v_maxtol, v_prune)
                            end
                        end
                    end
                end
                if prune_aux
                    error("Not implemented")
                end
                for l in lits_to_conjoin
                    new_n = conjoin(new_n, l)
                end
                new_n
            end
        end
    end
    f_o(_, cns)::Union{LogicCircuit,Nothing} = begin
        if all(cns .=== nothing)
            nothing
        else
            new_cns = filter(n -> n !== nothing, cns)
            disjoin(new_cns...)
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Union{LogicCircuit,Nothing})
end


"""
Find nodes that can be safely conjoined with an input auxiliary variable
"""
function aux_var_prune_candidates(lc::LogicCircuit, data::Matrix{Union{Bool,Missing}}; num_aux::Integer)
    num_vars = num_variables(lc) - 2 * num_aux

    ret_n = nothing
    ret_var = nothing
    ret_con = nothing

    # Check for each variable
    for var = Var(num_vars + 1) : Var(num_vars + num_aux)
        f_pos_cache = lc_flows(lc, data; pos_lit_missing_treatment = true, neg_lit_missing_treatment = false, applicable_vars = [var])
        f_neg_cache = lc_flows(lc, data; pos_lit_missing_treatment = false, neg_lit_missing_treatment = true, applicable_vars = [var])

        data_missing = ismissing.(data[:,var])

        foreach_down(lc) do n
            if isinner(n) && (ret_n === nothing)
                pos_flows = f_pos_cache[n]
                neg_flows = f_neg_cache[n]
                
                if !(any(pos_flows[.!data_missing]) || any(neg_flows[.!data_missing]))
                    if !any(pos_flows[data_missing]) && any(neg_flows[data_missing])
                        ret_n = n
                        ret_var = var
                        ret_con = true # Semantic: replace missing by true
                    elseif any(pos_flows[data_missing]) && !any(neg_flows[data_missing])
                        ret_n = n
                        ret_var = var
                        ret_con = false # Semantic: replace missing by true
                    end
                end
            end
        end
    end
    return ret_n, ret_var, ret_con
end


function post_prune_update_data(lc::LogicCircuit, data::Matrix{Union{Bool,Missing}}, data_sep::Vector{Bool}, var::Var; num_aux::Integer)
    v_pos_cache = lc_values(lc, data; pos_lit_missing_treatment = true, neg_lit_missing_treatment = false, applicable_vars = [var])
    v_neg_cache = lc_values(lc, data; pos_lit_missing_treatment = false, neg_lit_missing_treatment = true, applicable_vars = [var])
    pos_values = v_pos_cache[lc]
    neg_values = v_neg_cache[lc]
    for idx = 1 : size(data, 1)
        if ismissing(data[idx, var])
            if pos_values[idx] && !neg_values[idx]
                @inbounds data[idx, var] = true
                if idx > 1 && !data_sep[idx-1]
                    @inbounds data[idx-1, var+num_aux] = true # Set the corresponding output auxiliary var
                end
            elseif !pos_values[idx] && neg_values[idx]
                @inbounds data[idx, var] = false
                if idx > 1 && !data_sep[idx-1]
                    @inbounds data[idx-1, var+num_aux] = false # Set the corresponding output auxiliary var
                end
            end
        end
    end
end


function model_count_replace(lc::LogicCircuit, old2new::Dict{LogicCircuit,Union{LogicCircuit,Nothing}})
    cache = Dict{LogicCircuit,Int}()

    mcount(n) = begin
        if n === nothing
            zero(Int)
        else
            f_con(_)::Int = error("Does not support constant node")
            f_lit(_)::Int = one(Int)
            f_a(_, cns)::Int = prod(cns)
            f_o(_, cns)::Int = sum(cns)
            foldup_aggregate(n, f_con, f_lit, f_a, f_o, Int, cache)
        end
    end

    f_con(_)::Int = error("Does not support constant node")
    f_lit(n)::Int = haskey(old2new, n) ? mcount(old2new[n]) : one(Int)
    f_a(n, cns)::Int = haskey(old2new, n) ? mcount(old2new[n]) : prod(cns)
    f_o(n, cns)::Int = haskey(old2new, n) ? mcount(old2new[n]) : sum(cns)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Int)
end


function get_num_parents(lc::LogicCircuit)
    cache = Dict{LogicCircuit,Int}()
    foreach(lc) do n
        for c in children(lc)
            if !haskey(cache, c)
                cache[c] = 0
            end
            cache[c] += 1
        end
    end
    cache
end
