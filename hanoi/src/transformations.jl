# -*- coding: utf-8 -*-
using ProbabilisticCircuits


function deepcopy_lc(lc::LogicCircuit, depth::Int64; 
                     cache::Dict{LogicCircuit,LogicCircuit} = Dict{LogicCircuit,LogicCircuit}())
    if depth == 0 || isliteralgate(lc) || isconstantgate(lc)
        lc
    elseif haskey(cache, lc)
        cache[lc]
    else
        # Copy children
        cns = map(children(lc)) do c
            deepcopy_lc(c, depth - 1; cache)
        end
        if is⋀gate(lc)
            new_lc = conjoin(cns)
        else
            @assert is⋁gate(lc)
            new_lc = disjoin(cns)
        end
        cache[lc] = new_lc
        new_lc
    end
end


function split_lc(lc::LogicCircuit, (or, and)::Tuple{LogicCircuit,LogicCircuit}, var::Var; depth = 0)
    new_ch1 = map(children(and)) do c
        conjoin(c, var2lit(var))
    end
    new_ch2 = map(children(and)) do c
        conjoin(c, - var2lit(var))
    end

    new_and1 = deepcopy_lc(conjoin(new_ch1), depth)
    new_and2 = deepcopy_lc(conjoin(new_ch2), depth)
    new_or = disjoin([[new_and1, new_and2]; filter(c -> c != and, children(or))])

    replace_node(lc, or, new_or), new_or
end


"""
Prune away nodes in `candidates` from `lc`. The function will maintain smoothness.
"""
function prune_nodes(lc::LogicCircuit, candidates::Vector{LogicCircuit})
    f_con(_)::Union{LogicCircuit,Nothing} = error("Does not support constant nodes.")
    f_lit(n)::Union{LogicCircuit,Nothing} = n
    f_a(n, cns)::Union{LogicCircuit,Nothing} = begin
        if n in candidates
            nothing
        elseif any(cns .== nothing) 
            # If any child of an `and` node is set to false, then the whole `and` node should be set to false
            # This is the key to help maintain smoothness
            nothing
        else
            new_cns = filter(n -> n !== nothing, cns)
            conjoin(new_cns...)
        end
    end
    f_o(n, cns)::Union{LogicCircuit,Nothing} = begin
        if n in candidates
            nothing
        elseif all(cns .== nothing)
            # We prune `or` nodes only when all its children are pruned.
            nothing
        else
            new_cns = filter(n -> n !== nothing, cns)
            disjoin(new_cns...)
        end
    end
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, Union{LogicCircuit,Nothing})
end


"""
Convert a LC to a PC with identical structure
"""
function lc2pc(lc::LogicCircuit)
    f_con(_)::ProbCircuit = error("Does not support constant nodes.")
    f_lit(n)::ProbCircuit = PlainProbLiteralNode(n.literal)
    f_a(_, cns)::ProbCircuit = multiply(cns...)
    f_o(_, cns)::ProbCircuit = summate(cns...)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, ProbCircuit)
end


function standardize_lc(lc::LogicCircuit)
    f_con(n) = n
    f_lit(n) = n
    f_a(_, cns) = begin
        chs = map(cns) do c
            if is⋀gate(c)
                disjoin(c)
            else
                c
            end
        end
        conjoin(chs...)
    end
    f_o(_, cns) = disjoin(cns...)
    foldup_aggregate(lc, f_con, f_lit, f_a, f_o, LogicCircuit)
end
