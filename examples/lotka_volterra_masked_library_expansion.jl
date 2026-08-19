
# Closure-term recovery with an expanded candidate library.
#
# Same setup as lotka_volterra_masked_example.jl (Model 2 / mask2):
#   TRUE:    u1' = 1.5*u1 - u1*u2          u2' = 3*u1*u2 - u2
#   PARTIAL: u1' = 1.5*u1  (missing closure term u1*u2)
#            u2' = 3*u1*u2 - u2  (correct, known)
#
# Question: does the sampler still cleanly identify the true closure term
# u1*u2 (and reject the true absences) once the candidate library is
# widened with more distractor terms it has to sort through?
#
# Original library: full 3rd-order bivariate polynomial, no intercept (D=9).
# Expanded library: adds the complete 4th-order terms (D=14).
#
# Uses the tuned settings from lotka_volterra_masked_tuning.jl
# (nbasis=400, buffer=20, batch=50, kappa=1, scale_el=0.001), dense
# observations, R=0.1 additive noise — the same baseline regime as the
# original masked example, isolating the library-size effect.

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

Random.seed!(42)
R = 0.1 * Matrix{Float64}(I, 2, 2)
Z_tmp = copy(Y) + rand(MvNormal(zeros(2), R), nsamps)
Z = Matrix{Float64}([(Z_tmp[i, j] < 0 ? 0.0 : Z_tmp[i, j]) for i in 1:2, j in 1:nsamps])


###################### expanded library ######################

# Full 4th-order bivariate polynomial expansion, no intercept (D = 14):
#
#  Index | Term        | degree
#  ------+-------------+-------
#    1   |  u1         |  1
#    2   |  u2         |  1
#    3   |  u1*u2      |  2   <- true closure term
#    4   |  u1^2       |  2
#    5   |  u2^2       |  2
#    6   |  u1^2*u2    |  3
#    7   |  u1*u2^2    |  3
#    8   |  u1^3       |  3
#    9   |  u2^3       |  3
#   10   |  u1^3*u2    |  4
#   11   |  u1^2*u2^2  |  4
#   12   |  u1*u2^3    |  4
#   13   |  u1^4       |  4
#   14   |  u2^4       |  4

ΛNames = ["Phi"]

function Λ(A, Φ)
    ϕ = Φ[1]
    u = A * ϕ'
    X = u[1, :]
    Y = u[2, :]
    return [X, Y, X .* Y, X.^2, Y.^2, X.^2 .* Y, X .* Y.^2, X.^3, Y.^3,
            X.^3 .* Y, X.^2 .* Y.^2, X .* Y.^3, X.^4, Y.^4]
end

N, D = 2, 14


###################### mask (Model 2, same known terms/exclusions) ######################

mask2 = fill(-1, N, D)
mask2[1, 1] = 1;  mask2[2, 2] = 1;  mask2[2, 3] = 1  # known terms fixed in
mask2[1, 2] = 0;  mask2[2, 1] = 0                    # known absences fixed out


###################### tuned sampler settings ######################

nbasis        = 400
buffer        = 20
batch_size    = 50
learning_rate = 1.0
v0            = 1e-6
v1            = 1e4
nits          = 20000
scale_el      = fill(0.001, 2)


###################### run ######################

model, pars, posterior = DEtection_sampler(
    Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
    nits = nits, scale_el = scale_el, gamma_mask = mask2
)


###################### output ######################

println("=== Expanded library (D=14): known terms fixed, closure term free among more distractors ===")
print_equation(["u1_t", "u2_t"], model, pars, posterior, cutoff_prob = 0.5, p = 0.95)

post = posterior_summary(model, pars, posterior)

term_names = ["u1", "u2", "u1*u2", "u1^2", "u2^2", "u1^2*u2", "u1*u2^2", "u1^3", "u2^3",
              "u1^3*u2", "u1^2*u2^2", "u1*u2^3", "u1^4", "u2^4"]

println("\n--- inclusion probabilities (gamma) ---")
println(@sprintf("%-12s  %8s  %8s", "term", "eq1 (u1')", "eq2 (u2')"))
for (j, name) in enumerate(term_names)
    println(@sprintf("%-12s  %8.4f  %8.4f", name, post.gamma[1, j], post.gamma[2, j]))
end

println("\n--- posterior mean coefficients (M), closure term ---")
println(@sprintf("M[u1*u2, eq1] = %7.4f  (truth -1.0)", post.M[1, 3]))
println(@sprintf("M[u1*u2, eq2] = %7.4f  (truth  3.0)", post.M[2, 3]))

post_mean, post_sd = posterior_surface(model, pars, posterior)
inner = collect(model.inner_inds)
r_u1 = cor(post_mean[1, inner], Y[1, inner])
r_u2 = cor(post_mean[2, inner], Y[2, inner])
println(@sprintf("\nr_u1=%.3f  r_u2=%.3f", r_u1, r_u2))

p1 = plot(TimeStep, post_mean[1, :], ribbon = 2 .* post_sd[1, :], fillalpha = 0.2,
          label = "posterior mean ±2σ", color = :blue, title = "u1 — expanded library")
plot!(p1, TimeStep, Y[1, :], label = "truth", color = :black, lw = 1.5, linestyle = :dash)

p2 = plot(TimeStep, post_mean[2, :], ribbon = 2 .* post_sd[2, :], fillalpha = 0.2,
          label = "posterior mean ±2σ", color = :red, title = "u2 — expanded library")
plot!(p2, TimeStep, Y[2, :], label = "truth", color = :black, lw = 1.5, linestyle = :dash)

p_gamma = bar(term_names, [post.gamma[1, :] post.gamma[2, :]],
              label = ["eq1 (u1')" "eq2 (u2')"], xrotation = 45,
              title = "Inclusion probabilities, D=14 library", ylabel = "gamma")

p_summary = plot(plot(p1, p2, layout = (2, 1)), p_gamma, layout = (1, 2), size = (1300, 600))
display(p_summary)
savefig(p_summary, joinpath(@__DIR__, "../results/lv_library_expansion.png"))
println("\nPlot saved to results/lv_library_expansion.png")
