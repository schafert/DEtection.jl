
# Systematic parameter tuning for hare-lynx predator-prey example
# Goal: recover oscillating posterior mean matching Figure S5 in North et al. (2022)
#
# KNOWN STATE (from κ=1 run):
#   - κ=1 recovers oscillatory *shape* but amplitude is too small (damped)
#   - κ=10 (paper default) gives flat mean — gradient too large on iter 1 (ΣV=I init)
#
# AMPLITUDE PROBLEM DIAGNOSIS:
#   The elastic net in update_A!() (scale_el=1/100) shrinks A toward zero each step.
#   Over 10000 iterations this accumulated shrinkage damps amplitude even when shape
#   is correct.  Fixes to try, in priority order:
#     1. More iterations — A needs more steps to escape the shrinkage
#     2. Larger batch_size — reduces gradient variance, stabler convergence
#     3. Smaller κ — smaller steps allow more iterations before diverging
#     4. More buffer — reduces edge-effect contamination of gradient
#
# USAGE: run this file end-to-end. All configs run automatically.
#   Results saved to results/ for later inspection.
#   A summary comparison plot is shown at the end.

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using DataFrames, DataFramesMeta, Chain, CSV
using Printf
using Pipe: @pipe

mkpath("results")

###################### load data ######################
@pipe hare = "data/hare_lynx_data.csv" |>
             CSV.File |>
             DataFrame

@pipe Y = hare[:, 2:3] |> Matrix |> transpose |> copy

T_data = size(Y, 2)   # 91
years  = hare[:, :Year]

println("Data: $T_data time points, Hare $(round(minimum(Y[1,:]),digits=1))–$(round(maximum(Y[1,:]),digits=1)), ",
        "Lynx $(round(minimum(Y[2,:]),digits=1))–$(round(maximum(Y[2,:]),digits=1))")

###################### library ######################
ΛNames = ["Phi"]
function Λ(A, Φ)
  ϕ = Φ[1]
  u = A * ϕ'
  H = u[1,:]
  L = u[2,:]
  return [H, L, H .* L, H.^2, L.^2, H.^2 .* L, H .* L .^2, H.^3, L.^3]
end

###################### metrics ######################

function amplitude_ratio(post_mean, Y, inner_inds)
    # ratio of posterior mean amplitude to data amplitude (1.0 = perfect scale)
    ratio_hare = (maximum(post_mean[1, inner_inds]) - minimum(post_mean[1, inner_inds])) /
                 (maximum(Y[1, inner_inds])          - minimum(Y[1, inner_inds]))
    ratio_lynx = (maximum(post_mean[2, inner_inds]) - minimum(post_mean[2, inner_inds])) /
                 (maximum(Y[2, inner_inds])          - minimum(Y[2, inner_inds]))
    return ratio_hare, ratio_lynx
end

function fit_score(post_mean, Y, inner_inds)
    r_hare    = cor(post_mean[1, inner_inds], Y[1, inner_inds])
    r_lynx    = cor(post_mean[2, inner_inds], Y[2, inner_inds])
    rmse_hare = sqrt(mean((post_mean[1, inner_inds] .- Y[1, inner_inds]).^2))
    rmse_lynx = sqrt(mean((post_mean[2, inner_inds] .- Y[2, inner_inds]).^2))
    return (r_hare=r_hare, r_lynx=r_lynx, rmse_hare=rmse_hare, rmse_lynx=rmse_lynx)
end

###################### runner ######################

function run_config(label, lr, nits_val, buf, batch; seed=42, nbasis=40, scale_el=[0.01, 0.01])
    println("\n", "="^60)
    println("CONFIG: $label")
    println("  κ=$lr  nits=$nits_val  buffer=$buf  batch_size=$batch  nbasis=$nbasis  scale_el=$scale_el")
    println("="^60)
    Random.seed!(seed)

    TimeStep = Vector(range(0.1, T_data * 0.1, step = 0.1))

    model, pars, posterior = DEtection_sampler(
        Y, TimeStep,
        nbasis,
        buf,
        batch,
        Float64(lr),
        1e-6, 1e4,
        Λ, ΛNames,
        nits = nits_val,
        scale_el = scale_el
    )

    post_mean, post_sd = posterior_surface(model, pars, posterior)
    inner = collect(model.inner_inds)

    score = fit_score(post_mean, Y, inner)
    amp_h, amp_l = amplitude_ratio(post_mean, Y, inner)

    println("  r_hare=$(round(score.r_hare,digits=3))  r_lynx=$(round(score.r_lynx,digits=3))")
    println("  amplitude_ratio — Hare: $(round(amp_h,digits=3))  Lynx: $(round(amp_l,digits=3))")
    println("  (amplitude_ratio=1.0 means perfect scale; <1.0 means damped)")

    return (label=label, model=model, pars=pars, posterior=posterior,
            post_mean=post_mean, post_sd=post_sd, score=score,
            amp_hare=amp_h, amp_lynx=amp_l)
end

###################### experiment grid ######################
#
# All configs keep κ=1.  Columns: label, κ, nits, buf, batch, nbasis, scale_el
#
# scale_el = [hare_penalty, lynx_penalty]
#   default was hardcoded 0.01 for both.
#   smaller → less shrinkage → larger amplitude.
#   try asymmetric values if one species is more damped than the other.

configs = [
    # label                               κ    nits   buf batch nbasis  scale_el
    ("κ=1 sel=0.01 nits=20k",            1,  20000,   2,  20,   40,  [0.01, 0.01]),  # baseline ×2 iters
    ("κ=1 sel=0.001 nits=20k",           1,  20000,   2,  20,   40,  [0.001, 0.001]),  # 10× less shrinkage
    ("κ=1 sel=0.0001 nits=20k",          1,  20000,   2,  20,   40,  [0.0001, 0.0001]),  # 100× less shrinkage
    ("κ=1 sel=[0.001,0.01] nits=20k",    1,  20000,   2,  20,   40,  [0.001, 0.01]),  # asymmetric: less on hare
    ("κ=1 sel=[0.01,0.001] nits=20k",    1,  20000,   2,  20,   40,  [0.01, 0.001]),  # asymmetric: less on lynx
    ("κ=1 sel=0.001 nits=50k",           1,  50000,   2,  20,   40,  [0.001, 0.001]),  # best sel + more iters
]

results = []

for (label, lr, nits_val, buf, batch, nbasis, sel) in configs
    res = run_config(label, lr, nits_val, buf, batch; nbasis=nbasis, scale_el=sel)
    push!(results, res)
end

###################### summary plots ######################

println("\n\n", "="^60)
println("SUMMARY")
println("="^60)
println(@sprintf("%-40s  %6s  %6s  %6s  %6s", "Config", "r_H", "r_L", "amp_H", "amp_L"))
for r in results
    println(@sprintf("%-40s  %6.3f  %6.3f  %6.3f  %6.3f",
        r.label, r.score.r_hare, r.score.r_lynx, r.amp_hare, r.amp_lynx))
end

# find best result by r_hare + r_lynx
best = argmax([r.score.r_hare + r.score.r_lynx for r in results])
println("\nBest config: $(results[best].label)")

# panel plot: all posterior means vs data for hares
p_hare = plot(years, Y[1,:], color=:black, label="Observed", lw=1.5,
              title="Hare — all configs", xlabel="Year", ylabel="Population")
p_lynx = plot(years, Y[2,:], color=:black, label="Observed", lw=1.5,
              title="Lynx — all configs", xlabel="Year", ylabel="Population")

colors = [:blue, :red, :green, :orange, :purple, :brown]
for (i, r) in enumerate(results)
    plot!(p_hare, years, r.post_mean[1,:], label=r.label, color=colors[i], lw=1.2, alpha=0.8)
    plot!(p_lynx, years, r.post_mean[2,:], label=r.label, color=colors[i], lw=1.2, alpha=0.8)
end

p_summary = plot(p_hare, p_lynx, layout=(2,1), size=(1000, 700), legend=:outerright)
display(p_summary)
savefig(p_summary, "results/hare_lynx_comparison.png")
println("Comparison plot saved to results/hare_lynx_comparison.png")

# detailed plot of best config
best_res = results[best]
inner = collect(best_res.model.inner_inds)
p_best = plot(
    plot(years, best_res.post_mean[1,:], ribbon=2*best_res.post_sd[1,:], fillalpha=0.2,
         label="Posterior mean ±2σ", color=:blue, title="Hare — BEST: $(best_res.label)"),
    plot(years, best_res.post_mean[2,:], ribbon=2*best_res.post_sd[2,:], fillalpha=0.2,
         label="Posterior mean ±2σ", color=:red,  title="Lynx — BEST: $(best_res.label)"),
    layout=(2,1), size=(900, 600)
)
scatter!(p_best[1], years, Y[1,:], label="Observed", color=:black, markersize=3)
scatter!(p_best[2], years, Y[2,:], label="Observed", color=:black, markersize=3)
display(p_best)
savefig(p_best, "results/hare_lynx_best.png")
println("Best-config plot saved to results/hare_lynx_best.png")
