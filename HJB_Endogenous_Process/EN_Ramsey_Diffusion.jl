#==============================================================================
    Code for solving the Hamiltonian Jacboi Bellman for
	   an Ramsey Model with a diffusion process for capital

 	Based on  Matlab code from Ben Moll:
        http://www.princeton.edu/~moll/HACTproject.htm

        Updated to julia 1.0.0
==============================================================================#

using Distributions, Plots, LinearAlgebra, SparseArrays
using Random

Random.seed!(12364)

γ= 2.0 #gamma parameter for CRRA utility
ρ = 0.05 #the discount rate
α = 1/3 # the curvature of the production function (cobb-douglas)
δ = 0.05 # the depreciation rate
σ = 0.5 #variance term
n=0.51


#=============================================================================
 	k our capital follows a process that depends on
            dk = (s⋅f(k) - (δ-σ^2)k)dt - σ⋅k⋅dW_[t]
        Where W is a Wiener process
=============================================================================#
#K_starting point, for mean of z process

k_st = ((α)/(ρ+δ+n-σ^2))^(1/(1-α))

# create the grid for k
H = 100 #number of points on grid
k_min = 0.1*k_st # min value
k_max = 10*k_st # max value
k = LinRange(k_min, k_max, H)
dk = (k_max-k_min)/(H-1)
k_min_1=k_min
k_max_1=k_max


γ= 2.0 #gamma parameter for CRRA utility
ρ = 0.05 #the discount rate
α = 1/3 # the curvature of the production function (cobb-douglas)
δ = 0.05 # the depreciation rate
σ = 0.5 #variance term
n=0.51


#create matrices for k and z
kk = k*1

# use Ito's lemma to find the drift and variance of our optimization equation

max_it = 100
ε = 0.1^(6)
Δ = 1000

# set up all of these empty matrices
Vaf_1, Vab_1, c_1 = [zeros(H,1) for i in 1:3]

y = pdf.(Normal(.99, σ), k)
plot(k,y, grid=false,
		xlabel="k", ylabel="Probability",
		legend=false, color="purple", title="PDF of k")
png("PDF_k")

#============================================================================#
# Now it's time to solve the model, first put in a guess for the value function
v0 = (kk.^α).^(1-γ)/(1-γ)/ρ
v_1=v0

maxit= 30 #set number of iterations (only need 6 to converge)
dist = [] # set up empty array for the convergence criteria

for j = 1:maxit
    V=v_1

    #Now set up the forward difference

    Vaf_1[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vaf_1[H,:] .= (k_max.^α - (δ+n-σ^2).*k_max).^(-γ) # imposes a constraint

    #backward difference
    Vab_1[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
    Vab_1[1,:] .= (k_min.^α - (δ+n-σ^2).*k_min).^(-γ)

    #I_concave = Vab .> Vaf # indicator for whether the value function is concave

    # Consumption and savings functions
    cf = Vaf_1.^(-1/γ)
    drf =(1 .-(cf/kk.^α))*kk.^α-(δ+n-σ^2).*kk

    # consumption and saving backwards difference
    cb = Vab_1.^(-1.0/γ)
    drb = (1 .-(cb/k.^α))*kk.^α-(δ+n-σ^2).*kk
    #println(sb)
    #consumption and derivative of the value function at the steady state

    c0 = kk.^α - (δ+n-σ^2).*kk
    Va0 = c0.^(-γ)

    # df chooses between the forward or backward difference

    If = drf.>0 # positive drift will ⇒ forward difference
    Ib = drb.<0 # negative drift ⇒ backward difference
    I0=(1.0.-If-Ib) # at steady state

    Va_upwind = Vaf_1.*If + Vab_1.*Ib + Va0.*I0 # need to include SS term

    global c_1 = Va_upwind.^(-1/γ)
    u = (c_1.^(1-γ))/(1-γ)

    # Now to constuct the A matrix
    X = -min.(drb, 0)/dk + (σ.*kk/(2*dk))
    Y = -max.(drf, 0)/dk + min.(drb, 0)/dk - (σ.*kk/(dk))
    Z = max.(drf, 0)/dk + (σ.*kk/(2*dk))

    updiag = 0
    updiag =[updiag; Z[1:H-1,1]; 0]
    updiag =(updiag[:])

    centerdiag=reshape(Y, H, 1)
    centerdiag = (centerdiag[:]) # for tuples

   lowdiag = X[2:H, 1]
   lowdiag=(lowdiag)

   # spdiags in Matlab allows for automatic trimming/adding of zeros
       # spdiagm does not do this
  A = sparse(Diagonal(centerdiag))+ [zeros(1, H); sparse(Diagonal(lowdiag)) zeros(H-1,1)] + sparse(Diagonal(updiag))[2:end, 1:H] # trim first element

  B = (1/Δ + ρ)*sparse(I,H,H) - A

  u_stacked= reshape(u, H, 1)
  V_stacked = reshape(V,H, 1)

  b = u_stacked + (V_stacked./Δ)

  V_stacked = B\b

  V = reshape(V_stacked, H, 1)

  V_change = V-v_1

  global v_1= V
  # need push function to add to an already existing array
  push!(dist, findmax(abs.(V_change))[1])
  if dist[j].< ε
      println("Value Function Converged Iteration=")
      println(j)
      break
  end

end

# calculate the savings for kk
ss_1 = kk.^α - (δ+n-σ^2).*kk

# Plot the savings vs. k
plot(k, c_1, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Consumption Policies")
plot!(k, zeros(H,1), line=:dash, color=:black)
png("OptimalCons")


plot(k, ss_1, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Savings Policies")
plot!(k, zeros(H,1), line=:dash, color=:black)
png("OptimalSavings")


# Plot the interest rate
r_1 = α.*kk.^(α-1)
plot(k, r_1, grid=false,
		xlabel="k", ylabel="r(k)",
        xlims=(k_min,k_max),
		legend=false, title="Optimal Interest Rate")
png("OptimalInterest")

plot(k, v_1, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k_min,k_max),
		legend=false, title="")
png("Value_function_vs_k")

#==================== Now resolve the model with σ misspecified =======#

# create a random distribution to pull from
Dist = Bernoulli(.45)
T = 1000 # set the number of external time periods

# Add in the misspecification
σ_g =0.02
Σ_g=[σ_g]
# set up all of these empty matrices
Vaf_2, Vab_2, c_2 = [zeros(H,1) for i in 1:3]
Val_2 =[]
savings=[]

maxit= 30 #set number of iterations (only need 6 to converge)
for t in 1:T
    # Now it's time to solve the model, first put in a guess for the value function
    global v0 = (kk.^α).^(1-γ)/(1-γ)/ρ
    v_2=v0
    global dist = [] # needs to be in the time loop to work properly
    indicator = rand(Dist)
    for j = 1:maxit
        V=v_2

        #Now set up the forward difference

        Vaf_2[1:H-1,:] = (V[2:H, :] - V[1:H-1,:])/dk
        Vaf_2[H,:] .= (k_max.^α - (δ+n-σ_g^2).*k_max).^(-γ) # imposes a constraint

        #backward difference
        Vab_2[2:H,:] = (V[2:H, :] - V[1:H-1,:])/dk
        Vab_2[1,:] .= (k_min.^α - (δ+n-σ_g^2).*k_min).^(-γ)

        #I_concave = Vab .> Vaf # indicator for whether the value function is concave

        # Consumption and savings functions
        cf = Vaf_2.^(-1/γ)
        drf =(1 .-(cf/kk.^α))*kk.^α-(δ+n-σ_g^2).*kk

        # consumption and saving backwards difference
        cb = Vab_2.^(-1.0/γ)
        drb = (1 .-(cb/k.^α))*kk.^α-(δ+n-σ_g^2).*kk
        #println(sb)
        #consumption and derivative of the value function at the steady state

        c0 = kk.^α - (δ+n-σ_g^2).*kk
        Va0 = c0.^(-γ)

        # df chooses between the forward or backward difference

        If = drf.>0 # positive drift will ⇒ forward difference
        Ib = drb.<0 # negative drift ⇒ backward difference
        I0=(1.0.-If-Ib) # at steady state

        Va_upwind = Vaf_2.*If + Vab_2.*Ib + Va0.*I0 # need to include SS term

        global c_2 = Va_upwind.^(-1/γ)
        u = (c_2.^(1-γ))/(1-γ)

        # Now to constuct the A matrix
        X = -min.(drb, 0)/dk + (σ_g.*kk/(2*dk))
        Y = -max.(drf, 0)/dk + min.(drb, 0)/dk - (σ_g.*kk/(dk))
        Z = max.(drf, 0)/dk + (σ_g.*kk/(2*dk))

        updiag = 0
        updiag =[updiag; Z[1:H-1,1]; 0]
        updiag =(updiag[:])

        centerdiag=reshape(Y, H, 1)
        centerdiag = (centerdiag[:]) # for tuples

       lowdiag = X[2:H, 1]
       lowdiag=(lowdiag)

       # spdiags in Matlab allows for automatic trimming/adding of zeros
           # spdiagm does not do this
      A = sparse(Diagonal(centerdiag))+ [zeros(1, H); sparse(Diagonal(lowdiag)) zeros(H-1,1)] + sparse(Diagonal(updiag))[2:end, 1:H] # trim first element

      B = (1/Δ + ρ)*sparse(I,H,H) - A

      u_stacked= reshape(u, H, 1)
      V_stacked = reshape(V,H, 1)

      b = u_stacked + (V_stacked./Δ)

      V_stacked = B\b

      V = reshape(V_stacked, H, 1)

      V_change = V-v_2

      v_2= V
      # need push function to add to an already existing array
      push!(dist, findmax(abs.(V_change))[1])
      if dist[j].< ε
        push!(Val_2, v_2)
        break
      end
    end
    push!(Σ_g, σ_g)
    # calculate the savings for kk
    ss_2 = kk.^α - (δ+n-σ_g^2).*kk
    push!(savings, ss_2)
    # add in updating
    if indicator ==1
        global σ_g = σ_g + .01(σ-σ_g)
    end
    println(t)
end

# Plot the savings vs. k
plot(k[1:H-1], c_2[1:H-1], grid=false,
		xlabel="k", ylabel="c(k)",
        xlims=(k[1],k[end]),legend=:bottomright,
		label="Guess", title="Optimal Consumption Policies")
plot!(k[1:H-1],c_1[1:H-1], label="True Value", line=:dash)
png("OptimalCons")

#=
plot(k[1:H-1],cons[1][1:H-1], grid=false,
		xlabel="k", ylabel="c(k)",
        xlims=(k[1],k[end]),label="Initial", legend=:bottomright,
        title="Optimal Consumption Policies", color=:hotpink)
plot!(k[1:H-1],cons[500][1:H-1], label="500th period", color=:blue)
plot!(k[1:H-1],cons[end][1:H-1], label="10,000th period", color=:green)
plot!(k[1:H-1],c_1[1:H-1], label="Actual", line=:dot, color=:black)
png("OptimalCons_2")
=#

plot(k, savings[1], grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k[1],k[end]),label="Period 1",
        title="Optimal Savings Policies",
		color=:hotpink, legend=:topleft, line=:dashdot)
plot!(k, zeros(H,1), color=:black, label="")
plot!(k,savings[100], label="Period 100", color=:blue,line=:dot)
plot!(k,savings[500], label="Period 500", color=:green, line=:dashdotdot)
plot!(k,savings[1000], label="Period 1,000", color=:orange)
plot!(k,ss_1, label="True Value", line=:dash, color=:black)
png("OptimalSavings_2")

plot(k, savings, grid=false,
		xlabel="k", ylabel="s(k)",
        xlims=(k[1],k[end]),label="", title="Optimal Savings Policies")
plot!(k, zeros(H,1), line=:dash, color=:black, label="", legend=:topleft)
plot!(k,ss_1, color=:black, label="True Value")
png("OptimalSavings_All")

plot(k,Val_2[1], label="Period 1 ", color=:hotpink, line=:dashdot)
plot!(k,Val_2[100], label="Period 100", color=:blue, line=:dot)
plot!(k,Val_2[500], label="Period 500", color=:green, line=:dashdotdot)
plot!(k,Val_2[end], label="Period 1,000", color=:orange )
plot!(k,v_1, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k[1],k[end]), title="Value Functions",
        legend=:bottomright, color=:black, line=:dash,
        label="True Value")
png("Value_function_vs_k_2")


plot(k, Val_2, grid=false,
		xlabel="k", ylabel="V(k)",
		xlims=(k[1],k[end]), title="Value Functions",
		label="", legend=:bottomright)
plot!(k,v_1, label="True Value", color=:black, line=:dash)
png("Value_function_vs_k_all")


plot([1:T], Σ_g[2:end], grid=false,
		xlabel="time", ylabel="\$\\sigma\$",
		title="\$ \\textrm{Estimate of } \\sigma \\textrm{ over time}\$",
		label="Misspecfication", legend=:bottomright)
plot!([1:T],ones(T).*σ, label="True Value", color=:black, line=:dash)
png("sigma")
