
# Systematic parameter tuning for the masked Lotka-Volterra example.
#
# TRUE DYNAMICS:
#   u1'(t) = 1.5*u1 - u1*u2
#   u2'(t) = 3*u1*u2 - u2
#
# PARTIAL (ERRONEOUS) MODEL supplied to sampler:
#   ũ1'(t) = 1.5*u1           (missing closure term: -u1*u2)
#   ũ2'(t) = 3*u1*u2 - u2     (correct)
#
# KNOWN ISSUE (from baseline run):
#   - u1 coefficient in eq1 recovered as ~0.06 (truth: 1.5)
#   - closure term u1*u2 in eq1 has gamma inclusion prob ≈ 0 (should → 1.0)
#   - eq2 coefficients in right direction but underestimated
#   Likely cause: basis function fit (A) not capturing fast dynamics well enough.
#
# TUNING AXES (in priority order):
#   1. nbasis      — more basis functions → better derivative approximation
#   2. scale_el    — smaller → less A shrinkage → better amplitude recovery
#   3. nits        — more iterations → more time to converge
#   4. lr          — smaller → stabler A updates
#
# All configs use mask2 (known terms fixed + explicit exclusions) as it is
# the more constrained model and most likely to recover the closure term cleanly.

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using Printf

mkpath(joinpath(@__DIR__, "../results"))


###################### data simulation ######################

function lotka_volterra(X, pars)
    a, b, d, g = pars
    x, y = X
    return [a*x - b*x*y, d*x*y - g*y]
end

function rk4(fn, X, pars, h)
    k1 = h * fn(X, pars)
    k2 = h * fn(X + k1/2, pars)
    k3 = h * fn(X + k2/2, pars)
    k4 = h * fn(X + k3, pars)
    return X + (1/6)*(k1 + 2k2 + 2k3 + k4)
end

pars_true = [1.5, 1.0, 3.0, 1.0]
X0       = [1.0, 0.5]
h        = 0.05
end_time = 30.0
nsamps   = convert(Int, end_time / h)

Y = Array{Float64}(undef, 2, nsamps)
Y[:, 1] = X0
for i in 2:nsamps
    Y[:, i] = rk4(lotka_volterra, Y[:, i-1], pars_true, h)
end

Random.seed!(42)
R     = 0.1 * Matrix{Float64}(I, 2, 2)
Z_tmp = copy(Y) + rand(MvNormal(zeros(2), R), nsamps)
Z     = Matrix{Float64}([(Z_tmp[i,j] < 0 ? 0.0 : Z_tmp[i,j]) for i in 1:2, j in 1:nsamps])


###################### library ######################

# 3rd-order polynomial, no intercept (D = 9)
# Index: 1=u1, 2=u2, 3=u1*u2, 4=u1², 5=u2², 6=u1²*u2, 7=u1*u2², 8=u1³, 9=u2³

ΛNames = ["Phi"]

function Λ(A, Φ)
    ϕ = Φ[1]
    u = A * ϕ'
    X = u[1, :]
    Y = u[2, :]
    return [X, Y, X .* Y, X.^2, Y.^2, X.^2 .* Y, X .* Y.^2, X.^3, Y.^3]
end


###################### mask (Model 2) ######################

# fix gamma=1: u1 in eq1 (idx 1), u2 in eq2 (idx 2), u1*u2 in eq2 (idx 3)
# fix gamma=0: u2 in eq1 (idx 2), u1 in eq2 (idx 1)

N, D = 2, 9
mask2 = fill(-1, N, D)
mask2[1, 1] = 1;  mask2[2, 2] = 1;  mask2[2, 3] = 1
mask2[1, 2] = 0;  mask2[2, 1] = 0


###################### metrics ######################

# Ground truth coefficients for the three key terms
#   M[1,1] = u1 in eq1       → 1.5
#   M[1,3] = u1*u2 in eq1    → -1.0  (the closure term)
#   M[2,2] = u2 in eq2       → -1.0
#   M[2,3] = u1*u2 in eq2    → 3.0

function recovery_metrics(model, pars, posterior, Y_true)

    post      = posterior_summary(model, pars, posterior)
    post_mean, post_sd = posterior_surface(model, pars, posterior)
    inner     = collect(model.inner_inds)

    # closure term discovery
    closure_prob = post.gamma[1, 3]

    # key coefficient errors (posterior mean − truth)
    err_u1_eq1    = post.M[1, 1] - 1.5
    err_u1u2_eq1  = post.M[1, 3] - (-1.0)
    err_u2_eq2    = post.M[2, 2] - (-1.0)
    err_u1u2_eq2  = post.M[2, 3] - 3.0

    # fit quality vs noiseless truth
    r_u1   = cor(post_mean[1, inner], Y_true[1, inner])
    r_u2   = cor(post_mean[2, inner], Y_true[2, inner])
    rmse_u1 = sqrt(mean((post_mean[1, inner] .- Y_true[1, inner]).^2))
    rmse_u2 = sqrt(mean((post_mean[2, inner] .- Y_true[2, inner]).^2))

    return (
        closure_prob  = closure_prob,
        M_u1_eq1      = post.M[1, 1],
        M_u1u2_eq1    = post.M[1, 3],
        M_u2_eq2      = post.M[2, 2],
        M_u1u2_eq2    = post.M[2, 3],
        err_u1_eq1    = err_u1_eq1,
        err_u1u2_eq1  = err_u1u2_eq1,
        err_u2_eq2    = err_u2_eq2,
        err_u1u2_eq2  = err_u1u2_eq2,
        r_u1          = r_u1,
        r_u2          = r_u2,
        rmse_u1       = rmse_u1,
        rmse_u2       = rmse_u2,
        post_mean     = post_mean,
        post_sd       = post_sd,
    )
end


###################### runner ######################

function run_config(label, lr, nits_val, buf, batch, nbasis, sel; seed=42)

    println("\n", "="^65)
    println("CONFIG: $label")
    println("  κ=$lr  nits=$nits_val  buffer=$buf  batch=$batch  nbasis=$nbasis  scale_el=$sel")
    println("="^65)
    Random.seed!(seed)

    TimeStep = Vector(range(h, end_time, step = h))

    model, pars, posterior = DEtection_sampler(
        Z, TimeStep,
        nbasis, buf, batch,
        Float64(lr),
        1e-6, 1e4,
        Λ, ΛNames,
        nits      = nits_val,
        scale_el  = fill(sel, 2),
        gamma_mask = mask2
    )

    m = recovery_metrics(model, pars, posterior, Y)

    println(@sprintf "  closure prob (u1·u2 in eq1): %.4f  [truth: γ=1]" m.closure_prob)
    println(@sprintf "  M[u1,eq1]   = %7.4f  (truth  1.5,  err %+.4f)" m.M_u1_eq1   m.err_u1_eq1)
    println(@sprintf "  M[u1u2,eq1] = %7.4f  (truth -1.0,  err %+.4f)" m.M_u1u2_eq1 m.err_u1u2_eq1)
    println(@sprintf "  M[u2,eq2]   = %7.4f  (truth -1.0,  err %+.4f)" m.M_u2_eq2   m.err_u2_eq2)
    println(@sprintf "  M[u1u2,eq2] = %7.4f  (truth  3.0,  err %+.4f)" m.M_u1u2_eq2 m.err_u1u2_eq2)
    println(@sprintf "  r_u1=%.3f  r_u2=%.3f  rmse_u1=%.3f  rmse_u2=%.3f" m.r_u1 m.r_u2 m.rmse_u1 m.rmse_u2)

    return (label=label, model=model, pars=pars, posterior=posterior, metrics=m)
end


###################### experiment grid ######################

configs = [
    # label                              κ     nits   buf  batch  nbasis  scale_el
    ("baseline (from masked run)",      10,  10000,  20,   50,   200,  0.01  ),
    ("nbasis=400",                      10,  10000,  20,   50,   400,  0.01  ),
    ("sel=0.001",                       10,  10000,  20,   50,   200,  0.001 ),
    ("nbasis=400 sel=0.001",            10,  20000,  20,   50,   400,  0.001 ),
    ("nbasis=400 sel=0.001 κ=1",         1,  20000,  20,   50,   400,  0.001 ),
    ("nbasis=400 sel=0.0001 κ=1",        1,  20000,  20,   50,   400,  0.0001),
]

results = []

for (label, lr, nits_val, buf, batch, nbasis, sel) in configs
    res = run_config(label, lr, nits_val, buf, batch, nbasis, sel)
    push!(results, res)
end


###################### summary ######################

println("\n\n", "="^65)
println("SUMMARY")
println("="^65)
println(@sprintf("%-38s  %7s  %7s  %7s  %7s  %6s  %6s",
        "Config", "γ_clos", "M_u1", "M_u1u2", "M_u1u2", "r_u1", "r_u2"))
println(@sprintf("%-38s  %7s  %7s  %7s  %7s  %6s  %6s",
        "", "(eq1)", "(eq1)", "(eq1)", "(eq2)", "", ""))
println(@sprintf("%-38s  %7s  %7s  %7s  %7s  %6s  %6s",
        "", "[→1]", "[→1.5]", "[→-1]", "[→3]", "", ""))
println("-"^65)
for r in results
    m = r.metrics
    println(@sprintf("%-38s  %7.4f  %7.4f  %7.4f  %7.4f  %6.3f  %6.3f",
        r.label, m.closure_prob, m.M_u1_eq1, m.M_u1u2_eq1, m.M_u1u2_eq2,
        m.r_u1, m.r_u2))
end

best = argmax([r.metrics.closure_prob + r.metrics.r_u1 + r.metrics.r_u2 for r in results])
println("\nBest config: $(results[best].label)")


###################### plots ######################

colors = [:grey, :blue, :green, :orange, :red, :purple]

p_u1 = plot(Vector(range(h, end_time, step=h)), Y[1, :],
            color=:black, lw=1.5, label="truth", title="u1 — all configs")
p_u2 = plot(Vector(range(h, end_time, step=h)), Y[2, :],
            color=:black, lw=1.5, label="truth", title="u2 — all configs")

for (i, r) in enumerate(results)
    plot!(p_u1, Vector(range(h, end_time, step=h)), r.metrics.post_mean[1, :],
          label=r.label, color=colors[i], lw=1.2, alpha=0.8)
    plot!(p_u2, Vector(range(h, end_time, step=h)), r.metrics.post_mean[2, :],
          label=r.label, color=colors[i], lw=1.2, alpha=0.8)
end

p_summary = plot(p_u1, p_u2, layout=(2, 1), size=(1100, 700), legend=:outerright)
display(p_summary)
savefig(p_summary, joinpath(@__DIR__, "../results/lv_masked_tuning_comparison.png"))
println("Comparison plot saved to results/lv_masked_tuning_comparison.png")

# best config detailed plot
best_res = results[best]
t_vec    = Vector(range(h, end_time, step=h))

p_best = plot(
    plot(t_vec, best_res.metrics.post_mean[1, :],
         ribbon = 2 .* best_res.metrics.post_sd[1, :], fillalpha=0.2,
         label="posterior mean ±2σ", color=:blue,
         title="u1 — BEST: $(best_res.label)"),
    plot(t_vec, best_res.metrics.post_mean[2, :],
         ribbon = 2 .* best_res.metrics.post_sd[2, :], fillalpha=0.2,
         label="posterior mean ±2σ", color=:red,
         title="u2 — BEST: $(best_res.label)"),
    layout=(2, 1), size=(900, 600)
)
plot!(p_best[1], t_vec, Y[1, :], label="truth", color=:black, lw=1.5, linestyle=:dash)
plot!(p_best[2], t_vec, Y[2, :], label="truth", color=:black, lw=1.5, linestyle=:dash)
display(p_best)
savefig(p_best, joinpath(@__DIR__, "../results/lv_masked_tuning_best.png"))
println("Best-config plot saved to results/lv_masked_tuning_best.png")
