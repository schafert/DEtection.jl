using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, Chain, CSV
using Pipe: @pipe

###################### load data ######################
@pipe hare = "data/hare_lynx_data.csv" |>
             CSV.File |>
             DataFrame

@pipe Y = hare[:, 2:3] |> Matrix |> transpose |> copy

###################### library ######################
ΛNames = ["Phi"]
function Λ(A, Φ)
  ϕ = Φ[1]
  u = A * ϕ'
  H = u[1,:]
  L = u[2,:]
  return [H, L, H .* L, H.^2, L.^2, H.^2 .* L, H .* L .^2, H.^3, L.^3]
end

######################### run sampler #########################

Random.seed!(42)

nbasis      = 40
TimeStep    = Vector(range(0.1, size(hare, 1) * 0.1, step = 0.1))
batch_size  = 20
buffer      = 2
v0          = 1e-6
v1          = 1e4
learning_rate = 1.0

# scale_el controls the elastic net regularization per component [hare, lynx].
# Tuning sweep found 0.0001 gives r > 0.94 and amplitude ratios ~0.89/0.94.
# Try scale_el = [0.00001, 0.00001] to see if amplitude can be pushed closer to 1.0.
scale_el = [0.0001, 0.0001]

model, pars, posterior = DEtection_sampler(
    Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
    nits = 20000,
    scale_el = scale_el
)

model

######################### output #########################

try
    eq = print_equation(["Hₜ", "Lₜ"], model, pars, posterior, cutoff_prob = 0.5, p = 0.95)
    println(eq)
catch e
    println("print_equation failed (no terms selected at this cutoff): ", e)
end

post = posterior_summary(model, pars, posterior)

post_mean, post_sd = posterior_surface(model, pars, posterior)

plot(post_mean[1,:], label = "Posterior mean")
plot!(Y[1,:], label = "Observed")
title!("Hare")

plot(post_mean[2,:], label = "Posterior mean")
plot!(Y[2,:], label = "Observed")
title!("Lynx")
