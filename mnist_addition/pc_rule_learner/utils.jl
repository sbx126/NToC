
import LogicCircuits: pos_literals, neg_literals # extend

function pos_literals(::Type{T}, num_lits) where T <: ProbCircuit
    map(1:num_lits) do v
        PCs.PlainInputNode(v, Literal(true))
    end
end

function neg_literals(::Type{T}, num_lits) where T <: ProbCircuit
    map(1:num_lits) do v
        PCs.PlainInputNode(v, Literal(false))
    end
end

function cat_literals(::Type{T}, num_vars::Int, num_cats::Int; ret_type = "or") where T <: LogicCircuit
    pos_lits = pos_literals(T, num_vars * num_cats)
    neg_lits = neg_literals(T, num_vars * num_cats)

    cat_lits = Matrix{LogicCircuit}(undef, num_vars, num_cats)
    for v = 1 : num_vars
        for c = 1 : num_cats
            if ret_type == "or"
                n = disjoin(conjoin([ifelse(i == c, pos_lits[(v-1)*num_cats+i], neg_lits[(v-1)*num_cats+i]) for i = 1 : num_cats]...))
            elseif ret_type == "and"
                n = conjoin([ifelse(i == c, pos_lits[(v-1)*num_cats+i], neg_lits[(v-1)*num_cats+i]) for i = 1 : num_cats]...)
            else
                error("Unknown ret_type = $(ret_type).")
            end
            @inbounds cat_lits[v, c] = n
        end
    end
    cat_lits
end
function cat_literals(::Type{T}, num_cats::Vector{Int}; ret_type = "or") where T <: LogicCircuit
    num_lits = sum(num_cats)
    pos_lits = pos_literals(T, num_lits)
    neg_lits = neg_literals(T, num_lits)

    cat_lits = Vector{LogicCircuit}(undef, num_lits)
    v, c = 1, 1
    for l = 1 : num_lits
        lit_start = sum(num_cats[1:v-1]) + 1
        lit_end = sum(num_cats[1:v])
        chs = [ifelse(i == l, pos_lits[i], neg_lits[i]) for i = lit_start : lit_end]
        if ret_type == "or"
            n = disjoin(conjoin(chs...))
        elseif ret_type == "and"
            n = conjoin(chs...)
        else
            error("Unknown ret_type = $(ret_type).")
        end
        @inbounds cat_lits[l] = n

        if c == num_cats[v]
            c = 1
            v += 1
        else
            c += 1
        end
    end
    cat_lits
end
function cat_literals(::Type{T}, num_cats::Vector{Int}; ret_type = "or") where T <: ProbCircuit
    num_lits = sum(num_cats)
    pos_lits = pos_literals(T, num_lits)
    neg_lits = neg_literals(T, num_lits)

    cat_lits = Vector{ProbCircuit}(undef, num_lits)
    v, c = 1, 1
    for l = 1 : num_lits
        lit_start = sum(num_cats[1:v-1]) + 1
        lit_end = sum(num_cats[1:v])
        chs = [ifelse(i == l, pos_lits[i], neg_lits[i]) for i = lit_start : lit_end]
        if ret_type == "or"
            n = summate(multiply(chs...))
        elseif ret_type == "and"
            n = multiply(chs...)
        else
            error("Unknown ret_type = $(ret_type).")
        end
        @inbounds cat_lits[l] = n

        if c == num_cats[v]
            c = 1
            v += 1
        else
            c += 1
        end
    end
    cat_lits
end

function get_cat_lit(cat_lits::Vector{T}, num_cats::Vector{Int}, v::Int, c::Int) where T
    l = sum(num_cats[1:v-1]) + c
    cat_lits[l]
end

import ProbabilisticCircuits: loglikelihood

loglikelihood(d::Literal, value::Float32, _ = nothing) = ifelse(d.value, log(value), log(one(Float32) - value))