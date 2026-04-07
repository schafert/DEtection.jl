

using DEtection
using Plots, Missings, Distributions, Random, LinearAlgebra, Statistics


###################### data functions ######################

# True dynamics:
#   u1'(t) = 1.5*u1 - u1*u2
#   u2'(t) = 3*u1*u2 - u2
#
# Partial (erroneous) model known to ecologist:
#   ũ1'(t) = 1.5*u1             (missing -u1*u2)
#   ũ2'(t) = 3*u1*u2 - u2      (correct)

function lotka_volterra(X, pars)

    a = pars[1]  # prey growth
    b = pars[2]  # predation
    d = pars[3]  # predator growth from predation
    g = pars[4]  # predator death

    x = X[1]
    y = X[2]

    x_prime = a * x - b * x * y
    y_prime = d * x * y - g * y

    return Array([x_prime, y_prime])
end

function rk4(fn, X, pars, h)

    k1 = h * fn(X, pars)
    k2 = h * fn(X + k1 / 2, pars)
    k3 = h * fn(X + k2 / 2, pars)
    k4 = h * fn(X + k3, pars)

    return X + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

end


###################### create data ######################

pars_true = [1.5, 1.0, 3.0, 1.0]
X0 = [1.0, 0.5]
h = 0.05
end_time = 30.0
nsamps = convert(Int, end_time / h)

Y = Array{Float64}(undef, 2, nsamps)
Y[:, 1] = X0
for i in 2:nsamps
    Y[:, i] = rk4(lotka_volterra, Y[:, i-1], pars_true, h)
end

######################### noisy data #########################

Random.seed!(42)
R = 0.1 * Matrix{Float64}(I, 2, 2)

Z_tmp = copy(Y) + rand(MvNormal(zeros(2), R), nsamps)
Z = [(Z_tmp[i, j] < 0 ? 0.0 : Z_tmp[i, j]) for i in 1:2, j in 1:nsamps]
Z = Matrix{Float64}(Z)


###################### library ######################

# 3rd-order polynomial expansion, no intercept (D = 9):
#
#  Index | Term
#  ------+--------
#    1   |  u1
#    2   |  u2
#    3   |  u1*u2
#    4   |  u1^2
#    5   |  u2^2
#    6   |  u1^2*u2
#    7   |  u1*u2^2
#    8   |  u1^3
#    9   |  u2^3

ΛNames = ["Phi"]

function Λ(A, Φ)

    ϕ = Φ[1]
    u = A * ϕ'

    X = u[1, :]
    Y = u[2, :]

    return [X, Y, X .* Y, X.^2, Y.^2, X.^2 .* Y, X .* Y.^2, X.^3, Y.^3]

end


######################### sampler settings #########################

nbasis        = 400
TimeStep      = Vector(range(h, end_time, step = h))
batch_size    = 50
buffer        = 20
v0            = 1e-6
v1            = 1e4
learning_rate = 1e0

N = 2  # components (u1, u2)
D = 9  # library terms


######################### Model 1 #########################
# Fix gamma = 1 for the three terms present in the partial model:
#   eq1: u1       (index 1)  ← 1.5*u1 term
#   eq2: u2       (index 2)  ← -u2 term
#   eq2: u1*u2    (index 3)  ← 3*u1*u2 term
# All other entries are free (-1).

mask1 = fill(-1, N, D)
mask1[1, 1] = 1   # u1     always included in u1 equation
mask1[2, 2] = 1   # u2     always included in u2 equation
mask1[2, 3] = 1   # u1*u2  always included in u2 equation

model1, pars1, posterior1 = DEtection_sampler(
    Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
    nits = 20000, scale_el = fill(0.001, 2), gamma_mask = mask1)


######################### Model 2 #########################
# Same as Model 1, plus explicit exclusions based on partial knowledge:
#   eq1: u2 excluded (index 2) — predator count alone doesn't drive prey growth
#   eq2: u1 excluded (index 1) — prey count alone doesn't drive predator growth

mask2 = copy(mask1)
mask2[1, 2] = 0   # u2 excluded from u1 equation
mask2[2, 1] = 0   # u1 excluded from u2 equation

model2, pars2, posterior2 = DEtection_sampler(
    Z, TimeStep, nbasis, buffer, batch_size, learning_rate, v0, v1, Λ, ΛNames,
    nits = 20000, scale_el = fill(0.001, 2), gamma_mask = mask2)


######################### output #########################

println("=== Model 1: known terms fixed gamma=1, all others free ===")
print_equation(["u1_t", "u2_t"], model1, pars1, posterior1, cutoff_prob = 0.95, p = 0.95)

println("\n=== Model 2: known terms fixed gamma=1 + u2 excl. from eq1, u1 excl. from eq2 ===")
print_equation(["u1_t", "u2_t"], model2, pars2, posterior2, cutoff_prob = 0.95, p = 0.95)

post1 = posterior_summary(model1, pars1, posterior1)
post2 = posterior_summary(model2, pars2, posterior2)

# posterior mean coefficient matrices (rows = equations, cols = library terms)
println("\n--- Model 1: posterior mean M ---")
println(post1.M)
println("\n--- Model 2: posterior mean M ---")
println(post2.M)

# inclusion probabilities
println("\n--- Model 1: gamma inclusion probabilities ---")
println(post1.gamma)
println("\n--- Model 2: gamma inclusion probabilities ---")
println(post2.gamma)

# fitted vs truth
post_mean1, _ = posterior_surface(model1, pars1, posterior1)
post_mean2, _ = posterior_surface(model2, pars2, posterior2)

p1 = plot(post_mean1[1, :], label = "fitted u1", title = "Model 1 — u1")
plot!(p1, Y[1, :], label = "truth", linestyle = :dash)

p2 = plot(post_mean1[2, :], label = "fitted u2", title = "Model 1 — u2")
plot!(p2, Y[2, :], label = "truth", linestyle = :dash)

p3 = plot(post_mean2[1, :], label = "fitted u1", title = "Model 2 — u1")
plot!(p3, Y[1, :], label = "truth", linestyle = :dash)

p4 = plot(post_mean2[2, :], label = "fitted u2", title = "Model 2 — u2")
plot!(p4, Y[2, :], label = "truth", linestyle = :dash)

plot(p1, p2, p3, p4, layout = (2, 2), size = (900, 600))
