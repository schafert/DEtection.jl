
# Closure-term recovery vs. observation sparsity and additive noise.
#
# Same setup as lotka_volterra_masked_example.jl (Model 2 / mask2):
#   TRUE:    u1' = 1.5*u1 - u1*u2          u2' = 3*u1*u2 - u2
#   PARTIAL: u1' = 1.5*u1  (missing closure term u1*u2)
#            u2' = 3*u1*u2 - u2  (correct, known)
#
# mask2 fixes the known terms (gamma=1) and known absences (gamma=0),
# leaving the closure term u1*u2 in eq1 free to be discovered.
#
# This sweep holds the tuned sampler settings from lotka_volterra_masked_tuning.jl
# (nbasis=400, buffer=20, batch=50, kappa=1, scale_el=0.001) fixed and varies:
#   - sparsity: fraction of the 600 timepoints observed, thinned on a regular
#     grid (every k-th point kept, rest marked `missing`)
#   - noise: additive observation noise variance R
#
# tracking closure-term inclusion probability and coefficient recovery as the
# outcome of interest.

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics
using Printf

mkpath(joinpath(@__DIR__, "../results"))


###################### true dynamics ######################

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
X0        = [1.0, 0.5]
h         = 0.05
end_time  = 30.0
nsamps    = convert(Int, end_time / h)

Y = Array{Float64}(undef, 2, nsamps)
Y[:, 1] = X0
for i in 2:nsamps
    Y[:, i] = rk4(lotka_volterra, Y[:, i-1], pars_true, h)
end

TimeStep = Vector(range(h, end_time, step = h))


###################### library ######################

# Index: 1=u1, 2=u2, 3=u1*u2, 4=u1^2, 5=u2^2, 6=u1^2*u2, 7=u1*u2^2, 8=u1^3, 9=u2^3

ΛNames = ["Phi"]

function Λ(A, Φ)
    ϕ = Φ[1]
    u = A * ϕ'
    X = u[1, :]
    Y = u[2, :]
    return [X, Y, X .* Y, X.^2, Y.^2, X.^2 .* Y, X .* Y.^2, X.^3, Y.^3]
end


###################### mask (Model 2) ######################

N, D = 2, 9
mask2 = fill(-1, N, D)
mask2[1, 1] = 1;  mask2[2, 2] = 1;  mask2[2, 3] = 1  # known terms fixed in
mask2[1, 2] = 0;  mask2[2, 1] = 0                    # known absences fixed out


###################### fixed tuned sampler settings ######################

nbasis        = 400
buffer        = 20
batch_size    = 50
learning_rate = 1.0
v0            = 1e-6
v1            = 1e4
nits          = 20000
scale_el      = fill(0.001, 2)


###################### data generation ######################

function make_observations(seed, R_var, keep_every)
    Random.seed!(seed)
    R = R_var * Matrix{Float64}(I, 2, 2)
    Z_tmp = copy(Y) + rand(MvNormal(zeros(2), R), nsamps)
    Z_dense = [(Z_tmp[i, j] < 0 ? 0.0 : Z_tmp[i, j]) for i in 1:2, j in 1:nsamps]

    Z = Array{Union{Missing,Float64}}(missing, N, nsamps)
    keep_cols = 1:keep_every:nsamps
    Z[:, keep_cols] = Z_dense[:, keep_cols]
    return Z
end


###################### recovery metrics ######################

function recovery_metrics(model, pars, posterior, Y_true)
    post      = posterior_summary(model, pars, posterior)
    post_mean, post_sd = posterior_surface(model, pars, posterior)
    inner     = collect(model.inner_inds)

    closure_prob = post.gamma[1, 3]

    r_u1 = cor(post_mean[1, inner], Y_true[1, inner])
    r_u2 = cor(post_mean[2, inner], Y_true[2, inner])

    return (
        closure_prob = closure_prob,
        M_u1u2_eq1   = post.M[1, 3],
        err_u1u2_eq1 = post.M[1, 3] - (-1.0),
        r_u1         = r_u1,
        r_u2         = r_u2,
        post_mean    = post_mean,
        post_sd      = post_sd,
    )
end


###################### runner ######################

function run_config(label, R_var, keep_every; seed = 42)
    println("\n", "="^65)
    println("CONFIG: $label")
    frac_observed = length(1:keep_every:nsamps) / nsamps
    println(@sprintf("  R_var=%.3f  keep_every=%d  frac_observed=%.2f", R_var, keep_every, frac_observed))
    println("="^65)

    Z = make_observations(seed, R_var, keep_every)

    # Reseed before the sampler so each config's MCMC exploration draws from
    # an independent RNG stream. Without this, make_observations() always
    # consumes the same number of random draws regardless of R_var, so
    # DEtection_sampler starts every config at a fixed keep_every from a
    # byte-identical RNG state — at high sparsity, where the SSVS inclusion
    # step (update_gamma!) saturates to a near-deterministic function of that
    # shared stream, this silently locks gamma's trajectory in lockstep
    # across noise levels instead of letting it respond to the data. See
    # 2026-08-19_lv-closure-experiments.qmd (OneDrive) for the investigation.
    Random.seed!(seed + keep_every * 1000 + round(Int, R_var * 100))

    model, pars, posterior = DEtection_sampler(
        Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
        nits = nits, scale_el = scale_el, gamma_mask = mask2
    )

    m = recovery_metrics(model, pars, posterior, Y)

    println(@sprintf "  closure prob (u1*u2 in eq1): %.4f  [truth: gamma=1]" m.closure_prob)
    println(@sprintf "  M[u1u2,eq1] = %7.4f  (truth -1.0, err %+.4f)" m.M_u1u2_eq1 m.err_u1u2_eq1)
    println(@sprintf "  r_u1=%.3f  r_u2=%.3f" m.r_u1 m.r_u2)

    return (label = label, R_var = R_var, keep_every = keep_every,
            frac_observed = frac_observed, metrics = m)
end


###################### experiment grid ######################

noise_levels    = [0.05, 0.1, 0.3]   # additive observation noise variance
sparsity_levels = [1, 2, 5]          # keep every k-th timepoint

results = []
for R_var in noise_levels, keep_every in sparsity_levels
    label = @sprintf("R=%.2f, 1/%d pts", R_var, keep_every)
    push!(results, run_config(label, R_var, keep_every))
end


###################### summary ######################

println("\n\n", "="^65)
println("SUMMARY")
println("="^65)
println(@sprintf("%-20s  %7s  %7s  %7s  %6s  %6s",
        "Config", "frac", "gamma", "M_u1u2", "r_u1", "r_u2"))
println("-"^65)
for r in results
    m = r.metrics
    println(@sprintf("%-20s  %7.2f  %7.4f  %7.4f  %6.3f  %6.3f",
        r.label, r.frac_observed, m.closure_prob, m.M_u1u2_eq1, m.r_u1, m.r_u2))
end


###################### heatmaps ######################

closure_grid = [only([r.metrics.closure_prob for r in results
                       if r.R_var == rv && r.keep_every == ke])
                 for rv in noise_levels, ke in sparsity_levels]

err_grid = [only([abs(r.metrics.err_u1u2_eq1) for r in results
                   if r.R_var == rv && r.keep_every == ke])
            for rv in noise_levels, ke in sparsity_levels]

xticks_lab = ["1/$(k)" for k in sparsity_levels]
yticks_lab = ["$(rv)" for rv in noise_levels]

p_gamma = heatmap(xticks_lab, yticks_lab, closure_grid,
                   title = "Closure-term inclusion prob (truth=1)",
                   xlabel = "fraction of points kept", ylabel = "noise variance R",
                   clims = (0, 1), color = :viridis,
                   left_margin = 12Plots.mm, bottom_margin = 12Plots.mm)

p_err = heatmap(xticks_lab, yticks_lab, err_grid,
                 title = "abs error, M[u1*u2, eq1] (truth=-1.0)",
                 xlabel = "fraction of points kept", ylabel = "noise variance R",
                 color = :inferno,
                 left_margin = 12Plots.mm, bottom_margin = 12Plots.mm)

p_grid = plot(p_gamma, p_err, layout = (1, 2), size = (1300, 550), margin = 5Plots.mm)
display(p_grid)
savefig(p_grid, joinpath(@__DIR__, "../results/lv_sparsity_noise_grid.png"))
println("\nGrid heatmaps saved to results/lv_sparsity_noise_grid.png")
