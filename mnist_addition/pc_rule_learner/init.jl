

function generate_true_formula_boolean(num_vars, num_targets; type::String = "flat")
    pos_lits = pos_literals(LogicCircuit, num_vars + num_targets)
    neg_lits = neg_literals(LogicCircuit, num_vars + num_targets)

    if type == "flat"
        conj_children = map(1:num_vars+num_targets) do idx
            disjoin(pos_lits[idx], neg_lits[idx])
        end
        disjoin(conjoin(conj_children...))
    elseif type == "deep"
        lc = disjoin(pos_lits[1], neg_lits[1])
        for v = 2 : num_vars + num_targets
            conj1 = conjoin(lc, pos_lits[v])
            conj2 = conjoin(lc, neg_lits[v])
            lc = disjoin(conj1, conj2)
        end
        lc
    else
        error("Unknown LC type $(type).")
    end
end
function generate_true_formula_boolean(num_vars, num_targets, num_aux; type::String = "flag")
    total_num_vars = num_vars + num_targets + 2 * num_aux
    pos_lits = pos_literals(LogicCircuit, total_num_vars)
    neg_lits = neg_literals(LogicCircuit, total_num_vars)

    if type == "flat"
        conj_children = map(1:total_num_vars) do idx
            disjoin(pos_lits[idx], neg_lits[idx])
        end
        disjoin(conjoin(conj_children))
    elseif type == "deep"
        lc = disjoin(pos_lits[1], neg_lits[1])
        for v = 2 : num_vars + num_targets
            conj1 = conjoin(lc, pos_lits[v])
            conj2 = conjoin(lc, neg_lits[v])
            lc = disjoin(conj1, conj2)
        end
        for v = num_vars + num_targets + 1 : total_num_vars # Avoid the auxiliary variables to be "accidentally" splitted
            disj = disjoin(pos_lits[v], neg_lits[v])
            lc = disjoin(conjoin(lc, disj))
        end
        lc
    else
        error("Unknown LC type $(type).")
    end
end

function generate_true_formula_categorical(num_cats::Vector{Int}; type::String = "flat")
    num_vars = length(num_cats)
    cat_lits = cat_literals(LogicCircuit, num_cats; ret_type = "and")

    if type == "flat"
        conj_chs = map(1:num_vars) do v
            disjoin([get_cat_lit(cat_lits, num_cats, v, c) for c = 1 : num_cats[v]]...)
        end
        disjoin(conjoin(conj_chs))
    elseif type == "deep"
        lc = disjoin([get_cat_lit(cat_lits, num_cats, 1, c) for c = 1 : num_cats[1]]...)
        for v = 2 : num_vars
            conj_chs = map(1:num_cats[v]) do c
                conjoin(lc, get_cat_lit(cat_lits, num_cats, v, c))
            end
            lc = disjoin(conj_chs)
        end
        lc
    else
        error("Unknown LC type $(type).")
    end
end
