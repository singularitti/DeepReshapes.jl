# DeepReshapes

Extends
[reshape]{http://julia.readthedocs.org/en/latest/stdlib/base/#Base.reshape}
to arbitrarily nested structures of `Tuple`s and `Arrays`, both in source and
target.

## What?

Provides a `deep_reshape` function that can transform the structure of data:

```
A = [1 2; 3 4; 5 6]
b = [1, 2, 3, 4]
deep_reshape((A, b), (2, 5))
# => [1 5 4 1 3;
#     3 2 6 2 4]

deep_reshape([1:25], ((3, 3), (4, 4)))
# => ([ 1  4  7;
#       2  5  8;
#       3  6  9],
#     [10 14 18 22;
#      11 15 19 23;
#      12 16 20 24;
#      13 17 21 25])

α, β, c = deep_reshape([1.23, 2.34, 3, 4, 5], (Float64, Float64, (Int, 3)))
# => (1.23,2.34,[3,4,5])
```

This works on all (potentially nested) structures of `Tuple`s and `Arrays`, as
long as the actual scalar values contained within are `Number`s (for now).

## Why?

Say you want to optimize a non-linear function. Many optimization frameworks
([NLopt]{https://github.com/JuliaOpt/NLopt.jl},
[Optim]{https://github.com/JuliaOpt/Optim.jl}) require you to supply cost and
gradient functions and expect them to operate on plain `Vector{Float64}`s for
that purpose. However, some algorithms are expressed more naturally in terms of
more structured data.

Consider for example the popular
[backpropagation algorithm]
{http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm}
for training neural networks:

```
# (just for illustration:)
function cost_and_gradient!(
  W::Vector{Matrix{Float64}},  # weight matrices for each neuron layer
  b::Vector{Vector{Float64}},  # bias vectors for each neuron layer
  ∇W::Vector{Matrix{Float64}}, # vector to receive resulting weight gradients
  ∇b::Vector{Vector{Float64}}  # vector to receive resulting bias gradients
)
  # ...do feedforward and backpropagation, obtain some intermediate results
  # ...calculate gradients and fill the result vectors ∇W and ∇b
  # ...calculate and return the cost
end
```

For optimization, we cannot use this function directly, because it expects it to
work on plain number vectors:

```
using NLopt

W, b = initialize_parameters()
# ...we need to flatten W, b to number vector θ

optimization = Opt(:LD_LBFGS, length(θ))
min_objective!(optimization, cost_and_gradient!) # <- we need to define this
result = optimize(optimization, θ)
```

Flattening the initial parameters is easy with `DeepReshapes.flatten()`:

```
using DeepReshapes

θ = flatten(Float64, W, b)
```

As for `cost_and_gradient!`, we can simply wrap the original calculation with
`deep_reshape`s:

```
function cost_and_gradient!(θ::Vector{Float64}, ∇θ::Vector{Float64})
  W, b = deep_reshape(θ, s) # <- s is the original structure which can be
                            # obtained by calling DeepReshapes.structure() on
                            # the initial parameters before flattening them.

  # ...do the original calculation
  ∇θ[:] = flatten(Float64, ∇W, ∇b)
  # ... calculate and return the cost
end
```

---

_More to follow..._
