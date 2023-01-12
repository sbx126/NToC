# -*- coding: utf-8 -*-
using DataStructures


function folddown_aggregate(lc::LogicCircuit, f_con::Function, f_lit::Function, f_a::Function, f_o::Function, ::Type{T}, cache = nothing) where {T}
    node_parents::Dict{LogicCircuit,Vector{LogicCircuit}} = Dict{LogicCircuit,Vector{LogicCircuit}}()
    if cache === nothing
        cache = Dict{LogicCircuit, T}()
    end
    nodes_to_visit = Deque{LogicCircuit}()

    bfs(n) = begin
        if !haskey(cache, n)
            pn = map(node_parents[n]) do par
                cache[par]
            end
            if isleaf(n)
                cache[n] = isliteralgate(n) ? f_lit(n, pn) : f_con(n, pn)
            else
                cache[n] = is⋁gate(n) ? f_o(n, pn) : f_a(n, pn)
                for c in children(n)
                    if !(c in keys(cache))
                        if !haskey(node_parents, c)
                            node_parents[c] = [n]
                        else
                            push!(node_parents[c], n)
                        end
                        if !(c in nodes_to_visit)
                            push!(nodes_to_visit, c)
                        end
                    end
                end
            end
        end
    end

    node_parents[lc] = []
    push!(nodes_to_visit, lc)
    while length(nodes_to_visit) > 0
        n = popfirst!(nodes_to_visit)
        bfs(n)
    end

    cache
end


function folddown_each(f::Function, pc::ProbCircuit)
    node_parents::Dict{ProbCircuit,Vector{ProbCircuit}} = Dict{ProbCircuit,Vector{ProbCircuit}}()
    cache = Set{ProbCircuit}()
    nodes_to_visit = Deque{ProbCircuit}()

    bfs(n) = begin
        if !(n in cache)
            f(n)
            push!(cache, n)
            if isinner(n)
                for c in children(n)
                    if !(c in cache)
                        if !haskey(node_parents, c)
                            node_parents[c] = [n]
                        else
                            push!(node_parents[c], n)
                        end
                        if !(c in nodes_to_visit)
                            push!(nodes_to_visit, c)
                        end
                    end
                end
            end
        end
    end

    node_parents[pc] = []
    push!(nodes_to_visit, pc)
    while length(nodes_to_visit) > 0
        n = popfirst!(nodes_to_visit)
        bfs(n)
    end

    nothing
end

# import ProbabilisticCircuits: inputs

inputs(n::PCs.PlainSumNode) = n.inputs
inputs(n::PCs.PlainMulNode) = n.inputs

function folddown_each(f::Function, lc::LogicCircuit)
    node_parents::Dict{LogicCircuit,Vector{LogicCircuit}} = Dict{LogicCircuit,Vector{LogicCircuit}}()
    cache = Set{LogicCircuit}()
    nodes_to_visit = Deque{LogicCircuit}()

    bfs(n) = begin
        if !(n in cache)
            f(n)
            push!(cache, n)
            if isinner(n)
                for c in children(n)
                    if !(c in cache)
                        if !haskey(node_parents, c)
                            node_parents[c] = [n]
                        else
                            push!(node_parents[c], n)
                        end
                        if !(c in nodes_to_visit)
                            push!(nodes_to_visit, c)
                        end
                    end
                end
            end
        end
    end

    node_parents[lc] = []
    push!(nodes_to_visit, lc)
    while length(nodes_to_visit) > 0
        n = popfirst!(nodes_to_visit)
        bfs(n)
    end

    nothing
end
