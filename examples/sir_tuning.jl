
# Systematic parameter tuning for the SIR example (no masking — full discovery).
#
# TRUE DYNAMICS, in population FRACTIONS s,i,r ∈ [0,1] (s+i+r = 1, so N=1
# and no /N normalization is needed):
#   s'(t) = -β*s*i
#   i'(t) =  β*s*i - γ*i
#   r'(t) =  γ*i
#
# with β=2.0, γ=0.5 (R0=4), giving a full rise/peak/decay epidemic curve
# over 30 time units.
#
# NOTE: an earlier version of this script worked in raw population counts
# (N=100), which divides the S*I coefficient down to ∓0.02 — two orders of
# magnitude smaller than the O(1-3) coefficients the tuned scale_el/v0/v1
# settings (carried over from the LV closure example) were calibrated for.
# Every config in that version fit the state trajectories perfectly (basis
# smoothing) but never selected the S*I term (gamma≈0) in either equation —
# a coefficient-scale mismatch, not a real modeling failure. Working in
# fractions puts coefficients back in the O(1) regime.
#
# Library is the same 3rd-order polynomial in (s, i) used previously (r does
# not appear — it doesn't drive any of the three equations).
#
# GOAL: recover s*i in eq1 (coeff -β), s*i and i in eq2, i in eq3,
# with all other terms correctly excluded (gamma → 0).

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using Printf

mkpath(joinpath(@__DIR__, "../results"))


###################### data simulation ######################

function sir(X, pars)
    β, γ = pars
    s, i, r = X
    return [-β*s*i, β*s*i - γ*i, γ*i]
end

function rk4(fn, X, pars, h)
    k1 = h*fn(X,pars); k2 = h*fn(X+k1/2,pars); k3 = h*fn(X+k2/2,pars); k4 = h*fn(X+k3,pars)
    return X + (1/6)*(k1+2k2+2k3+k4)
end

pars_true = [2.0, 0.5]   # β, γ
X0        = [0.99, 0.01, 0.0]
h         = 0.05
end_time  = 30.0
nsamps    = convert(Int, end_time / h)

Y = Array{Float64}(undef, 3, nsamps)
Y[:, 1] = X0
for i in 2:nsamps
    Y[:, i] = rk4(sir, Y[:, i-1], pars_true, h)
end

println("Peak I = $(round(maximum(Y[2,:]),digits=3)) at t=$(round(argmax(Y[2,:])*h,digits=2)); final S=$(round(Y[1,end],digits=3)), R=$(round(Y[3,end],digits=3))")

TimeStep = Vector(range(h, end_time, step = h))

Random.seed!(42)
R_noise = 0.001 * Matrix{Float64}(I, 3, 3)
Z_tmp = copy(Y) + rand(MvNormal(zeros(3), R_noise), nsamps)
Z = Matrix{Float64}([(Z_tmp[i, j] < 0 ? 0.0 : Z_tmp[i, j]) for i in 1:3, j in 1:nsamps])


###################### library ######################

# Index: 1=S, 2=I, 3=S*I, 4=S^2, 5=I^2, 6=S^2*I, 7=S*I^2, 8=S^3, 9=I^3

ΛNames = ["Phi"]

function Λ(A, Φ)
    ϕ = Φ[1]
    u = A * ϕ'
    S = u[1, :]
    I = u[2, :]
    return [S, I, S .* I, S.^2, I.^2, S.^2 .* I, S .* I.^2, S.^3, I.^3]
end


###################### metrics ######################

# ground truth coefficients (only nonzero terms), fraction scale:
#   M[1,3] = s*i in eq1 (s') = -β = -2.0
#   M[2,3] = s*i in eq2 (i') =  β =  2.0
#   M[2,2] = i   in eq2 (i') = -γ = -0.5
#   M[3,2] = i   in eq3 (r') =  γ =  0.5

function recovery_metrics(model, pars, posterior, Y_true)
    post = posterior_summary(model, pars, posterior)
    post_mean, post_sd = posterior_surface(model, pars, posterior)
    inner = collect(model.inner_inds)

    err_SI_eq1 = post.M[1, 3] - (-2.0)
    err_SI_eq2 = post.M[2, 3] - 2.0
    err_I_eq2  = post.M[2, 2] - (-0.5)
    err_I_eq3  = post.M[3, 2] - 0.5

    r_S = cor(post_mean[1, inner], Y_true[1, inner])
    r_I = cor(post_mean[2, inner], Y_true[2, inner])
    r_R = cor(post_mean[3, inner], Y_true[3, inner])

    return (
        gamma_SI_eq1 = post.gamma[1, 3], gamma_SI_eq2 = post.gamma[2, 3],
        gamma_I_eq2  = post.gamma[2, 2], gamma_I_eq3  = post.gamma[3, 2],
        err_SI_eq1 = err_SI_eq1, err_SI_eq2 = err_SI_eq2,
        err_I_eq2 = err_I_eq2, err_I_eq3 = err_I_eq3,
        r_S = r_S, r_I = r_I, r_R = r_R,
        post_mean = post_mean, post_sd = post_sd,
    )
end


###################### runner ######################

function run_config(label, lr, nits_val, buf, batch, nbasis, sel, v0, v1; seed=42)
    println("\n", "="^65)
    println("CONFIG: $label")
    println("  κ=$lr  nits=$nits_val  buffer=$buf  batch=$batch  nbasis=$nbasis  scale_el=$sel  v0=$v0  v1=$v1")
    println("="^65)
    Random.seed!(seed)

    model, pars, posterior = DEtection_sampler(
        Z, TimeStep, nbasis, buf, batch, Float64(lr), v0, v1, Λ, ΛNames,
        nits = nits_val, scale_el = fill(sel, 3)
    )

    m = recovery_metrics(model, pars, posterior, Y)

    println(@sprintf("  gamma[S*I,eq1]=%.3f  gamma[S*I,eq2]=%.3f  gamma[I,eq2]=%.3f  gamma[I,eq3]=%.3f", m.gamma_SI_eq1, m.gamma_SI_eq2, m.gamma_I_eq2, m.gamma_I_eq3))
    println(@sprintf("  err SI,eq1=%+.4f  SI,eq2=%+.4f  I,eq2=%+.4f  I,eq3=%+.4f", m.err_SI_eq1, m.err_SI_eq2, m.err_I_eq2, m.err_I_eq3))
    println(@sprintf("  r_S=%.3f  r_I=%.3f  r_R=%.3f", m.r_S, m.r_I, m.r_R))

    return (label=label, metrics=m)
end


###################### experiment grid ######################

configs = [
    # label                     κ    nits    buf batch nbasis  scale_el   v0    v1
    ("baseline",               10,  10000,   10,  20,   200,   0.01,   1e-8, 1e2 ),
    ("nbasis=400",             10,  10000,   20,  50,   400,   0.01,   1e-6, 1e4 ),
    ("nbasis=400 sel=0.001",   10,  10000,   20,  50,   400,   0.001,  1e-6, 1e4 ),
    ("nbasis=400 sel=0.001 κ=1", 1,  20000,   20,  50,   400,   0.001,  1e-6, 1e4 ),
    ("nbasis=400 sel=0.0001 κ=1", 1, 20000,   20,  50,   400,   0.0001, 1e-6, 1e4 ),
]

results = []
for (label, lr, nits_val, buf, batch, nbasis, sel, v0, v1) in configs
    push!(results, run_config(label, lr, nits_val, buf, batch, nbasis, sel, v0, v1))
end


###################### summary ######################

println("\n\n", "="^65)
println("SUMMARY")
println("="^65)
println(@sprintf("%-28s  %6s  %6s  %6s  %6s  %6s  %6s  %6s",
        "Config", "gSI_1", "gSI_2", "gI_2", "gI_3", "r_S", "r_I", "r_R"))
for r in results
    m = r.metrics
    println(@sprintf("%-28s  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f",
        r.label, m.gamma_SI_eq1, m.gamma_SI_eq2, m.gamma_I_eq2, m.gamma_I_eq3, m.r_S, m.r_I, m.r_R))
end

score(m) = m.gamma_SI_eq1 + m.gamma_SI_eq2 + m.gamma_I_eq2 + m.gamma_I_eq3 + m.r_S + m.r_I + m.r_R
best = argmax([score(r.metrics) for r in results])
println("\nBest config: $(results[best].label)")


###################### plots ######################

t_vec = TimeStep
best_res = results[best]
p1 = plot(t_vec, best_res.metrics.post_mean[1, :], ribbon = 2 .* best_res.metrics.post_sd[1, :],
          fillalpha = 0.2, label = "posterior mean ±2σ", color = :blue, title = "S — BEST: $(best_res.label)")
plot!(p1, t_vec, Y[1, :], label = "truth", color = :black, lw = 1.5, linestyle = :dash)

p2 = plot(t_vec, best_res.metrics.post_mean[2, :], ribbon = 2 .* best_res.metrics.post_sd[2, :],
          fillalpha = 0.2, label = "posterior mean ±2σ", color = :red, title = "I — BEST: $(best_res.label)")
plot!(p2, t_vec, Y[2, :], label = "truth", color = :black, lw = 1.5, linestyle = :dash)

p3 = plot(t_vec, best_res.metrics.post_mean[3, :], ribbon = 2 .* best_res.metrics.post_sd[3, :],
          fillalpha = 0.2, label = "posterior mean ±2σ", color = :green, title = "R — BEST: $(best_res.label)")
plot!(p3, t_vec, Y[3, :], label = "truth", color = :black, lw = 1.5, linestyle = :dash)

p_best = plot(p1, p2, p3, layout = (3, 1), size = (900, 800))
display(p_best)
savefig(p_best, joinpath(@__DIR__, "../results/sir_tuning_best.png"))
println("Best-config plot saved to results/sir_tuning_best.png")
