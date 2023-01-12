# -*- coding: utf-8 -*-
using Printf: @printf
import ProbabilisticCircuits as PCs

include("./pc_rule_learner/rule_learner.jl")


function gen_recursive_add_dataset(base::Integer = 2, seq_len::Integer = 3; Float = Float64)
    base_convert(num::Integer, base::Integer, k::Integer) = begin
        bits = Vector{Int64}(undef, k)
        for i = 1 : k
            bits[i] = (num % base)
            num = num ÷ base
        end
        bits
    end
    
    data = zeros(Float, seq_len * base^(2*seq_len), base * 3)
    i = 1
    for num1 = 0 : base^seq_len - 1
        for num2 = 0 : base^seq_len - 1
            converted1 = base_convert(num1, base, seq_len)
            converted2 = base_convert(num2, base, seq_len)
            converted3 = base_convert(num1 + num2, base, seq_len)
            for j = 1 : seq_len
                @inbounds data[i+j-1,converted1[j]+1] = 1.0
                @inbounds data[i+j-1,converted2[j]+1+base] = 1.0
                @inbounds data[i+j-1,converted3[j]+1+2*base] = 1.0
            end
            i += seq_len
        end
    end
    
    data_sep = Vector{Bool}(undef, seq_len * base^(2*seq_len))
    for idx = 1 : size(data_sep, 1)
        if idx % seq_len == 0
            data_sep[idx] = true
        else
            data_sep[idx] = false
        end
    end
    
    data, data_sep
end


function add_auxiliary_variables(data, data_sep; num_auxs, Float = Float64)
    n_examples = size(data, 1)
    n_features = size(data, 2)
    
    data_aux = Matrix{Float}(undef, n_examples, n_features + 2 * num_auxs)
    @inbounds @views data_aux[:, 1:n_features] .= data
    @inbounds @views data_aux[:, n_features+1:end] .= 0.5
    for idx = 1 : n_examples
        if idx == 1 || data_sep[idx-1]
            @inbounds data_aux[idx, n_features+1] = 1.0
            @inbounds data_aux[idx, n_features+2:n_features+num_auxs] .= 0.0
        end
    end
    
    data_aux
end


function get_tautology_pc(base; num_auxs)
    num_cats = vcat([base for i = 1 : 3], [num_auxs for i = 1 : 2])
    cat_lits = cat_literals(ProbCircuit, num_cats)
    
    get_sum_lit(v) = begin
        summate([get_cat_lit(cat_lits, num_cats, v, i).inputs[1] for i = 1 : num_cats[v]]...)
    end
    
    chs = Vector{ProbCircuit}()
    for x1 = 1 : base
        for x2 = 1 : base
            prod_chs = Vector{ProbCircuit}()
            push!(prod_chs, get_cat_lit(cat_lits, num_cats, 1, x1))
            push!(prod_chs, get_cat_lit(cat_lits, num_cats, 2, x2))
            
            sum_chs = Vector{ProbCircuit}()
            for y = 1 : base
                n = multiply(
                    get_cat_lit(cat_lits, num_cats, 3, y),
                    get_sum_lit(4),
                    get_sum_lit(5)
                )
                push!(sum_chs, n)
            end
            push!(prod_chs, summate(sum_chs...))
            
            n = multiply(prod_chs...)
            push!(chs, n)
        end
    end
    
    pc = summate(chs...)
    
    init_parameters(pc; perturbation = 0.5)
    fix_pc_input_weights!(pc)
    
    pc
end


function fix_pc_input_weights!(pc::ProbCircuit)
    @inbounds @views pc.params .= log(1.0 / num_inputs(pc))
    for i = 1 : num_inputs(pc)
        @inbounds @views pc.inputs[i].inputs[3].params .= log(1.0 / num_inputs(pc.inputs[i].inputs[3]))
    end
end
function fix_pc_input_weights2!(pc::ProbCircuit)
    @inbounds @views pc.params .= log(1.0 / num_inputs(pc))
end


function get_marupdate_nodes(pc::ProbCircuit)
    nodes = Vector{ProbCircuit}()
    for i = 1 : num_inputs(pc)
        push!(nodes, pc.inputs[i].inputs[3])
    end
    push!(nodes, pc)
    
    nodes
end


function update_aux_variables(aux_vars, condprob1, condprob2; base, lr = 1.0)
    batch_size = size(aux_vars, 1)
    seq_len = size(aux_vars, 2)
    num_auxs = size(aux_vars, 3) ÷ 2
    
    condprob1 = reshape(condprob1, batch_size, seq_len, 2 * num_auxs)
    condprob2 = reshape(condprob2, batch_size, seq_len, base + 2 * num_auxs)
    
    for j = 2 : seq_len
        aux_vars[:,j,1:num_auxs] .= (1-lr) .* aux_vars[:,j,1:num_auxs] .+ lr .* 
            condprob1[:,j,1:num_auxs] ./ (2.0 .* condprob2[:,j,base+1:base+num_auxs])
        aux_vars[:,j,1:num_auxs] .= clamp.(aux_vars[:,j,1:num_auxs], 0.0, 1.0)
        aux_vars[:,j-1,num_auxs+1:2*num_auxs] .= aux_vars[:,j,1:num_auxs]
        if j == seq_len
            @assert all(isapprox.(aux_vars[:,j,num_auxs+1:2*num_auxs], 0.5; atol = 1e-3))
        end
    end
    
    aux_vars
end


function update_aux_variables_adam(aux_vars, aux_vars_m, aux_vars_v, num_iters, grads; lr = 0.1,
                                   beta1 = 0.9, beta2 = 0.999)
    batch_size = size(aux_vars, 1)
    seq_len = size(aux_vars, 2)
    num_auxs = size(aux_vars, 3) ÷ 2
    
    num_iters .+= 1
    for j = 2 : seq_len
        aux_vars_m[:,j,1:num_auxs] .= beta1 .* aux_vars_m[:,j,1:num_auxs] .+ (1.0 - beta1) .* grads[:,j,1:num_auxs]
        aux_vars_v[:,j,1:num_auxs] .= beta2 .* aux_vars_v[:,j,1:num_auxs] .+ (1.0 - beta2) .* (grads[:,j,1:num_auxs].^2)
        m_hat = aux_vars_m[:,j,1:num_auxs] ./ reshape(1.0 .- beta1.^num_iters, batch_size)
        v_hat = aux_vars_v[:,j,1:num_auxs] ./ reshape(1.0 .- beta2.^num_iters, batch_size)
        
        aux_vars[:,j,1:num_auxs] .+= lr .* m_hat ./ (sqrt.(v_hat) .+ 1e-8)
        aux_vars[:,j,1:num_auxs] .= clamp.(aux_vars[:,j,1:num_auxs], 0.0, 1.0)
        aux_vars[:,j-1,num_auxs+1:2*num_auxs] .= aux_vars[:,j,1:num_auxs]
        if j == seq_len
            @assert all(isapprox.(aux_vars[:,j,num_auxs+1:2*num_auxs], 0.5; atol = 1e-3))
        end
    end
    
    aux_vars, aux_vars_m, aux_vars_v, num_iters
end


function print_pc(pc::ProbCircuit, base::Integer; num_auxs::Integer)
    get_cat_id(n::ProbCircuit) = begin
        for i = 1 : num_inputs(n)
            if n.inputs[i].literal > zero(Lit)
                return lit2var(n.inputs[i].literal)
            end
        end
        return zero(Var)
    end
    
    println("====== PC params ======")
    for i = 1 : base^2
        println("> Child #$(i): $(get_cat_id(pc.inputs[i].inputs[1].inputs[1]) - 1) ", 
                "$(get_cat_id(pc.inputs[i].inputs[2].inputs[1]) - base - 1)")
        n = pc.inputs[i].inputs[3]
        for j = 1 : base
            print("  - Y = $(j-1) ")
            @printf("(Pr = %.2f):\n", exp(n.params[j]))
            print("    > Cin  = ")
            for k = 1 : num_auxs
                @printf("%.2f, ", exp(n.inputs[j].inputs[2].params[k]))
            end
            println("")
            print("    > Cout = ")
            for k = 1 : num_auxs
                @printf("%.2f, ", exp(n.inputs[j].inputs[3].params[k]))
            end
            println("")
        end
    end
end
