
# Closure-term discovery on the REAL Hudson's Bay hare-lynx data (1845-1935).
#
# Earlier hare-lynx work (hare_lynx_best.jl) does unconstrained discovery with
# a plain polynomial library on the real data, and the masked-LV work
# (lotka_volterra_masked_example.jl etc.) validates the closure-term
# machinery on SYNTHETIC data with a known-true answer. This script combines
# both: it applies the closure-term (partial-knowledge) framework to the
# REAL data, where the true interaction functional form is genuinely
# unknown up front.
#
# PARTIAL/KNOWN MODEL (structural, not magnitude — coefficients are still
# estimated from data):
#   H'(t) = a*H + [predation loss, unknown form]
#   L'(t) =        [predation gain, unknown form] - g*L
# i.e. the intrinsic growth term for hares and intrinsic death term for lynx
# are fixed in (gamma=1); self-limitation and all interaction candidates are
# left free for the sampler to discover. Bare cross-species terms (a bare L
# in H', a bare H in L') are fixed OUT (gamma=0): under mass-action predation
# there's no mechanism for one species to affect the other without an
# encounter, i.e. without both densities appearing in the term together.
# (Same reasoning already used for mask2 in lotka_volterra_masked_example.jl.)
#
# THE INTERACTION LIBRARY:
#   Simple LV assumes a Type I (linear) functional response: predation rate
#   proportional to H*L. Real predator-prey systems usually saturate
#   (Holling Type II: rate ~ H*L/(1+c*H)). The Type II form is nonlinear IN
#   THE PARAMETER c, so it can't be handed to this sampler directly (library
#   terms must be fixed functions of state with only a linear coefficient).
#
#   Instead we fix c at a few ecologically plausible values spanning the
#   observed hare-density range (from the data's own quartiles) and include
#   H*L/(1+c*H) at each fixed c as a SEPARATE library column — turning "does
#   the data prefer a saturating response, and how strong" into a model-
#   selection question over a small dictionary, which SSVS can answer.
#   Alongside these we keep the full polynomial library (through cubic),
#   whose higher powers of H*L act as a polynomial (Taylor) proxy for
#   saturation curvature if the fixed-c dictionary doesn't capture it well.

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, Chain, CSV
using Pipe: @pipe
using Printf

mkpath(joinpath(@__DIR__, "../results"))

###################### load data ######################

@pipe hare = "data/hare_lynx_data.csv" |> CSV.File |> DataFrame
@pipe Y = hare[:, 2:3] |> Matrix |> transpose |> copy

# half-saturation constants from the data's own hare-density quartiles:
#   c = 1/H* means predation rate reaches half its max at hare density H*
Hq = quantile(hare.Hare, [0.25, 0.5, 0.75])
c_vals = 1.0 ./ Hq   # [weak/late sat., medium sat., strong/early sat.]
println("Half-saturation hare densities (quartiles): ", round.(Hq, digits=1))
println("Corresponding c values: ", round.(c_vals, digits=4))


###################### library ######################

# Index: 1=H, 2=L, 3=H*L, 4=H^2, 5=L^2, 6=H^2*L, 7=H*L^2, 8=H^3, 9=L^3,
#        10=H*L/(1+c1*H), 11=H*L/(1+c2*H), 12=H*L/(1+c3*H)  [saturating dict.]

ΛNames = ["Phi"]

function Λ(A, Φ)
    ϕ = Φ[1]
    u = A * ϕ'
    H = u[1, :]
    L = u[2, :]
    c1, c2, c3 = c_vals
    return [H, L, H .* L, H.^2, L.^2, H.^2 .* L, H .* L.^2, H.^3, L.^3,
            (H .* L) ./ (1 .+ c1 .* H),
            (H .* L) ./ (1 .+ c2 .* H),
            (H .* L) ./ (1 .+ c3 .* H)]
end

N, D = 2, 12


###################### mask: structural growth/death terms only ######################

mask = fill(-1, N, D)
mask[1, 1] = 1   # intrinsic hare growth term structurally present
mask[2, 2] = 1   # intrinsic lynx death term structurally present
mask[1, 2] = 0   # bare L excluded from H' — no mass-action encounter without H
mask[2, 1] = 0   # bare H excluded from L' — no mass-action encounter without L
# everything else (self-limitation, all interaction candidates) left free


###################### sampler settings ######################
# starting point: the tuned real-data settings from hare_lynx_best.jl

Random.seed!(42)

nbasis        = 40
TimeStep      = Vector(range(0.1, size(hare, 1) * 0.1, step = 0.1))
batch_size    = 20
buffer        = 2
v0            = 1e-6
v1            = 1e4
learning_rate = 1.0
scale_el      = [0.0001, 0.0001]
nits          = 20000


###################### run ######################

model, pars, posterior = DEtection_sampler(
    Y, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
    nits = nits, scale_el = scale_el, gamma_mask = mask
)


###################### output ######################

term_names = ["H", "L", "H*L", "H^2", "L^2", "H^2*L", "H*L^2", "H^3", "L^3",
              "HL/(1+c1 H)", "HL/(1+c2 H)", "HL/(1+c3 H)"]

post = posterior_summary(model, pars, posterior)

println("\n--- inclusion probabilities (gamma) ---")
println(@sprintf("%-14s  %8s  %8s", "term", "eq1 (H')", "eq2 (L')"))
for (j, name) in enumerate(term_names)
    println(@sprintf("%-14s  %8.4f  %8.4f", name, post.gamma[1, j], post.gamma[2, j]))
end

println("\n--- posterior mean coefficients (M) ---")
for (j, name) in enumerate(term_names)
    println(@sprintf("%-14s  M[eq1]=%9.5f  M[eq2]=%9.5f", name, post.M[1, j], post.M[2, j]))
end

try
    eq = print_equation(["Hₜ", "Lₜ"], model, pars, posterior, cutoff_prob = 0.5, p = 0.95)
    println("\n", eq)
catch e
    println("print_equation failed (no terms selected at this cutoff): ", e)
end

post_mean, post_sd = posterior_surface(model, pars, posterior)
inner = collect(model.inner_inds)
r_H = cor(post_mean[1, inner], Y[1, inner])
r_L = cor(post_mean[2, inner], Y[2, inner])
println(@sprintf("\nr_H=%.3f  r_L=%.3f", r_H, r_L))

years = hare.Year
p1 = plot(years, post_mean[1, :], ribbon = 2 .* post_sd[1, :], fillalpha = 0.2,
          label = "posterior mean ±2σ", color = :blue, title = "Hare — real data closure fit")
scatter!(p1, years, Y[1, :], label = "observed", color = :black, markersize = 2)

p2 = plot(years, post_mean[2, :], ribbon = 2 .* post_sd[2, :], fillalpha = 0.2,
          label = "posterior mean ±2σ", color = :red, title = "Lynx — real data closure fit")
scatter!(p2, years, Y[2, :], label = "observed", color = :black, markersize = 2)

p_gamma = bar(term_names, [post.gamma[1, :] post.gamma[2, :]],
              label = ["eq1 (H')" "eq2 (L')"], xrotation = 45,
              title = "Inclusion probabilities", ylabel = "gamma",
              bottom_margin = 15Plots.mm)

p_summary = plot(plot(p1, p2, layout = (2, 1)), p_gamma, layout = (1, 2), size = (1300, 600))
display(p_summary)
savefig(p_summary, joinpath(@__DIR__, "../results/hare_lynx_closure.png"))
println("\nPlot saved to results/hare_lynx_closure.png")
