using LogicCircuits
using ProbabilisticCircuits
using LoopVectorization

#include("io.jl")
include("utils.jl")

include("init.jl")

include("logic_utils.jl")
include("logic_queries.jl")
include("pc_queries.jl")

include("transformations.jl")

include("gradients.jl")
include("cond_prob.jl")

#include("hard_learner/hard_learning_plain.jl")
#include("hard_learner/hard_learning_recursive.jl")
#include("hard_learner/hard_learning_cat_plain.jl")

#include("soft_learner/soft_learning_plain.jl")
#include("soft_learner/soft_learning_recursive.jl")
