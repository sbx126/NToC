include("./src/LearnLogicalRules.jl")


function hanoi_meta_state_optimal_policy(pillar_cmps, last_action; num_disks = 3)
    action_mapping = [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    action_dict = Dict{Tuple{Int,Int},Int}()
    action_dict[(1,2)] = 1
    action_dict[(1,3)] = 2
    action_dict[(2,1)] = 3
    action_dict[(2,3)] = 4
    action_dict[(3,1)] = 5
    action_dict[(3,2)] = 6

    smallest_pillar = begin
        if pillar_cmps[1]
            pillar_cmps[3] ? 3 : 1
        else
            pillar_cmps[2] ? 2 : 3
        end
    end

    medium_pillar = begin
        if smallest_pillar == 1
            pillar_cmps[2] ? 2 : 3
        elseif smallest_pillar == 2
            pillar_cmps[3] ? 3 : 1
        else
            pillar_cmps[1] ? 1 : 2
        end
    end

    largest_pillar = begin
        if smallest_pillar == 1
            pillar_cmps[2] ? 3 : 2
        elseif smallest_pillar == 2
            pillar_cmps[3] ? 1 : 3
        else
            pillar_cmps[1] ? 2 : 1
        end
    end

    last_action = action_mapping[last_action]
    if smallest_pillar == last_action[2]
        action = (medium_pillar, largest_pillar)
    else
        if num_disks % 2 == 1
            act2 = smallest_pillar - 1
        else
            act2 = smallest_pillar + 1
        end
        if act2 == 0 
            act2 = 3
        elseif act2 == 4
            act2 = 1
        end
        action = (smallest_pillar, act2)
    end

    action_dict[action]
end

function get_hanoi_pc(; groundtruth_weights::Bool = false, eps = 1e-2)
    #num_cats = [2, 2, 2, 6, 6]
    num_cats = [2, 2, 2, 2, 6, 6]
    cat_lits = cat_literals(ProbCircuit, num_cats)

    gen_full_cat_lit(v) = begin
        lits = [get_cat_lit(cat_lits, num_cats, v, c).inputs[1] for c = 1 : num_cats[v]]
        summate(lits)
    end
    
    map_compare(v) = begin
        v == 1 && return true, false, true
        v == 2 && return true, true, false
        v == 3 && return false, true, true
        v == 4 && return true, false, false
        v == 5 && return false, true, false
        v == 6 && return false, false, true
    end
    

    
    chs = Vector{ProbCircuit}()
    for cat1 = 1 : 6
        comp = map_compare(cat1)
        for cat2 = 1 : 6
            cat3 = hanoi_meta_state_optimal_policy(comp, cat2)
        
            if groundtruth_weights && cat3 > 0
                target_n = gen_full_cat_lit(5)
                @inbounds @views target_n.params .= Float32(eps)
                target_n.params[cat3] = one(Float32)
                target_n.params ./= sum(target_n.params)
                target_n.params .= log.(target_n.params)
                n = multiply(
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, cat2), 
                    target_n
                )
            else
                n = multiply( 
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, cat2),
                    gen_full_cat_lit(5)
                )
            end
            push!(chs, n)
        end
    end
    pc = summate(chs...)

    pc
end

function get_hanoi_aux_pc(; groundtruth_weights::Bool = false, eps = 1e-2)
    num_cats = [2, 2, 2, 6, 6, 6]
    cat_lits = cat_literals(ProbCircuit, num_cats)

    gen_full_cat_lit(v) = begin
        lits = [get_cat_lit(cat_lits, num_cats, v, c).inputs[1] for c = 1 : num_cats[v]]
        summate(lits)
    end
    
    map_compare(v) = begin
        v == 1 && return true, false, true
        v == 2 && return true, true, false
        v == 3 && return false, true, true
        v == 4 && return true, false, false
        v == 5 && return false, true, false
        v == 6 && return false, false, true
    end
    
    chs = Vector{ProbCircuit}()
    for cat1 = 1 : 6
        comp = map_compare(cat1)
        for cat2 = 1 : 6
            cat3 = hanoi_meta_state_optimal_policy(comp, cat2)
        
            if groundtruth_weights && cat3 > 0
                target_n = gen_full_cat_lit(5)
                @inbounds @views target_n.params .= Float32(eps)
                target_n.params[cat3] = one(Float32)
                target_n.params ./= sum(target_n.params)
                target_n.params .= log.(target_n.params)
                n = multiply(
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, cat2), 
                    target_n,
                    gen_full_cat_lit(6)
                )
            else
                n = multiply( 
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, cat2),
                    # gen_full_cat_lit(4),
                    gen_full_cat_lit(5),
                    gen_full_cat_lit(6)
                )
            end
            push!(chs, n)
        end
    end
    pc = summate(chs...)

    pc
end

function get_hanoi_aux_pc_fix(; groundtruth_weights::Bool = false, eps = 1e-2)
    num_cats = [2, 2, 2, 6, 6, 6]
    cat_lits = cat_literals(ProbCircuit, num_cats)

    gen_full_cat_lit(v) = begin
        lits = [get_cat_lit(cat_lits, num_cats, v, c).inputs[1] for c = 1 : num_cats[v]]
        summate(lits)
    end
    
    chs = Vector{ProbCircuit}()
    for cat1 = 1 : 2
        for cat2 = 1 : 2
            for cat3 = 1 : 2
                for cat4 = 1 : 6
                    cat5 = cat1 + cat2
                    
                    if groundtruth_weights && cat5 > 0
                        target_n = gen_full_cat_lit(5)
                        @inbounds @views target_n.params .= Float32(eps)
                        target_n.params[cat5] = one(Float32)
                        target_n.params ./= sum(target_n.params)
                        target_n.params .= log.(target_n.params)
                        n = multiply(
                            get_cat_lit(cat_lits, num_cats, 1, cat1),
                            get_cat_lit(cat_lits, num_cats, 2, cat2),
                            get_cat_lit(cat_lits, num_cats, 3, cat3),
                            get_cat_lit(cat_lits, num_cats, 4, cat4), 
                            target_n,
                            gen_full_cat_lit(6)
                        )
                    else
                        n = multiply( 
                            get_cat_lit(cat_lits, num_cats, 1, cat1),
                            get_cat_lit(cat_lits, num_cats, 2, cat2),
                            get_cat_lit(cat_lits, num_cats, 3, cat3),
                            get_cat_lit(cat_lits, num_cats, 4, cat4),
                            # gen_full_cat_lit(4),
                            gen_full_cat_lit(5),
                            gen_full_cat_lit(6)
                        )
                    end
                    push!(chs, n)
                end
            end
        end
    end
    pc = summate(chs...)

    pc
end

function get_hanoi_color_aux_pc(; groundtruth_weights::Bool = false, eps = 1e-2)
    num_cats = [2, 2, 2, 2, 2, 2, 6, 6, 6]
    cat_lits = cat_literals(ProbCircuit, num_cats)

    gen_full_cat_lit(v) = begin
        lits = [get_cat_lit(cat_lits, num_cats, v, c).inputs[1] for c = 1 : num_cats[v]]
        summate(lits)
    end
    
    map_compare(v) = begin
        v == 1 && return false,false,false,false,false,false
        v == 2 && return false,false,false,false,false,true
        v == 3 && return false,false,false,true,false,false
        v == 4 && return false,false,false,false,true,true
        v == 5 && return false,false,false,true,true,false
        v == 6 && return false,false,false,true,true,true
        v == 7 && return true,false,false,false,false,false
        v == 8 && return true,false,false,false,false,true
        v == 9 && return false,true,false,true,false,false
        v == 10 && return false,false,true,false,true,true
        v == 11 && return false,true,false,true,true,false
        v == 12 && return false,false,true,true,true,true
        v == 13 && return true,true,false,false,false,false
        v == 14 && return true,false,true,false,false,true
        v == 15 && return true,true,false,true,false,false
        v == 16 && return true,false,true,false,true,true
        v == 17 && return false,true,true,true,true,false
        v == 18 && return false,true,true,true,true,true
        v == 19 && return true,true,true,false,false,false
        v == 20 && return true,true,true,false,false,true
        v == 21 && return true,true,true,true,false,false
        v == 22 && return true,true,true,false,true,true
        v == 23 && return true,true,true,true,true,false
        v == 24 && return true,true,true,true,true,true
    end
    
    chs = Vector{ProbCircuit}()
    for cat1 = 1 : 24
        comp = map_compare(cat1)
        for cat2 = 1 : 6
            cat3 = hanoi_meta_state_optimal_policy(comp, cat2)
        
            if groundtruth_weights && cat3 > 0
                target_n = gen_full_cat_lit(8)
                @inbounds @views target_n.params .= Float32(eps)
                target_n.params[cat3] = one(Float32)
                target_n.params ./= sum(target_n.params)
                target_n.params .= log.(target_n.params)
                n = multiply(
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, comp[4] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 5, comp[5] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 6, comp[6] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 7, cat2), 
                    target_n,
                    gen_full_cat_lit(8)
                )
            else
                n = multiply( 
                    get_cat_lit(cat_lits, num_cats, 1, comp[1] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 2, comp[2] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 3, comp[3] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 4, comp[4] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 5, comp[5] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 6, comp[6] ? 2 : 1),
                    get_cat_lit(cat_lits, num_cats, 7, cat2),
                    # gen_full_cat_lit(4),
                    gen_full_cat_lit(8),
                    gen_full_cat_lit(9)
                )
            end
            push!(chs, n)
        end
    end
    pc = summate(chs...)

    pc
end

function fix_pc_input_weights(pc::ProbCircuit; target_scope::BitSet = BitSet([13,14,15,16,17,18]))
    cache = Dict{ProbCircuit,BitSet}()
    foreach(pc) do n
        s = pc_scope(n; cache)
        if issum(n) && length(n.params) > 1 && !issubset(s, target_scope)
            nch = length(n.params)
            n.params .= log(one(Float32) / nch)
            if n === pc
                # println("aaa")
            else
                # println("bbb")
            end
        elseif issum(n) && length(n.params) > 1
            # println("ccc")
        end
    end
    nothing
end

function fix_pc_input_weights2(pc::ProbCircuit; target_scope1::BitSet = BitSet([13,14,15,16,17,18]), target_scope2::BitSet = BitSet([19,20,21,22,23,24]))
    cache = Dict{ProbCircuit,BitSet}()
    foreach(pc) do n
        s = pc_scope(n; cache)
        if issum(n) && length(n.params) > 1 && !issubset(s, target_scope1) && !issubset(s, target_scope2)
            nch = length(n.params)
            n.params .= log(one(Float32) / nch)
        end
    end
    nothing
end

function fix_pc_input_weights3(pc::ProbCircuit; target_scope1::BitSet = BitSet([19,20,21,22,23,24]), target_scope2::BitSet = BitSet([25,26,27,28,29,30]))
    cache = Dict{ProbCircuit,BitSet}()
    foreach(pc) do n
        s = pc_scope(n; cache)
        if issum(n) && length(n.params) > 1 && !issubset(s, target_scope1) && !issubset(s, target_scope2)
            nch = length(n.params)
            n.params .= log(one(Float32) / nch)
        end
    end
    nothing
end

cu_bit_circuit = PCs.CuBitsProbCircuit;

function groundtruth_hanoi_dataset()

    map_compare(v) = begin
        v == 1 && return true, false, true
        v == 2 && return true, true, false
        v == 3 && return false, true, true
        v == 4 && return true, false, false
        v == 5 && return false, true, false
        v == 6 && return false, false, true
    end

    dataset = Matrix{Float32}(undef, 0, 18)
    for cat1 = 1 : 6
        comp = map_compare(cat1)
        for cat2 = 1 : 6
            cat3 = hanoi_meta_state_optimal_policy(comp, cat2)

            v1, v2, v3 = map_compare(cat1)
            sample = zeros(Float32, 18)
            if v1
                sample[2] = 1.0
            else
                sample[1] = 1.0
            end
            if v2
                sample[4] = 1.0
            else
                sample[3] = 1.0
            end
            if v3
                sample[6] = 1.0
            else
                sample[5] = 1.0
            end

            sample[6+cat2] = 1.0
            sample[12+cat3] = 1.0

            dataset = vcat(dataset, reshape(sample, (1, :)))
        end
    end
    dataset
end

function eval_pc(pc)
    mean(pc_values(CuBitsProbCircuit(pc), groundtruth_hanoi_dataset()))
end

function get_rat_pc(num_vars = 18)
    RAT(num_vars; num_nodes_region = 16, num_nodes_leaf = 16, rg_depth = 3, rg_replicas = 4)
end
