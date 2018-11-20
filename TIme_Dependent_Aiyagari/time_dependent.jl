#==============================================================================

This is code for a time dependent Aiyagari model that has a poisson process
determining state of wealth

This model also includes an MIT shock

Translated Julia code from Matlab code by Ben Moll:
	http://www.princeton.edu/~moll/HACTproject.htm

==============================================================================#

using Parameters, Distributions, Plots, Base.BLAS

@with_kw type Model_parameters
    γ= 2#gamma parameter for CRRA utility
    ρ = 0.05 #the discount rate
	δ = 0.05 #the depreciation rate
    α = 1/3 # the curvature of the production function (cobb-douglas)
    A_prod = 0.1 #the value of A in the production function
	z_1 = 1 #state value
	z_2 = 2*z_1 #value of state 2
	z= [z_1 z_2] #our state vector
	λ_1 = 1/3 #poisson intensity of state 1
	λ_2 = 1/3 #poisson intensity of state 2
	λ = [λ_1 λ_2] #our state intensity vector
	z_avg = (z_1*λ_2 + z_2*λ_1)/(λ_1+λ_2) # average z value
end

Param=Model_parameters()
@unpack_Model_parameters(Param)

T=200
N=400
dt=T/N
time = [0:N-1]*dt
time = convert(Array, time[1])'
max_price_it = 300
ε = 1e-5 # change later to -6???
relax = 0.1

# construct the sequence that defines TFP

corr =0.8
ν = 1-corr
A_prod_t = zeros(N,1)
A_prod_t[1] = .97*A_prod
ssf =0

for n=1:N-1
	A_prod_t[n+1] = dt*ν*(A_prod-A_prod_t[n]) + A_prod_t[n]
end

plot(time', A_prod_t, xlims=(0,40), legend=false)

#Now to look at a

H = 1000 # using H instead of I for easier updating with Julia 1.0.0
a_min =-0.8
a_max = 20
a = linspace(a_min, a_max, H)
a = convert(Array, a)
da = (a_max-a_min)/(H-1)

aa = [a a]
zz = ones(H,1)*z

maxit=1
ϵ = 1e-6
Δ = 1000

dVf = zeros(H,2)
dVb = zeros(H,2)
c = zeros(H,2)
V=zeros(H,2)
X, Y, Z = [zeros(H,1) for i in 1:3]
# this matrix tracks the evolution of a from the discretized HJB
A_switch = [-speye(H)*λ[1] speye(H)*λ[1]; speye(H)*λ[2] -speye(H)*λ[2]]

I_r = 40
crit_S =1e-5


r_max = 0.049
r=0.04
w =0.05

v_0 = zeros(H,2)
v=zeros(H,2)
v_0[:,1] = (w*z[1] + r.*a).^(1-γ)/(1-γ)/ρ
v_0[:,2] = (w*z[2] + r.*a).^(1-γ)/(1-γ)/ρ


r0=0.03
r_min = 0.01
r_max=0.99*ρ



# ========= First we need to caclulate the steady state =================#

# set up some matrices
r_r, r_minr, r_maxr, KD = [zeros(I_r) for i in 1:4]
V_n = []
dist = []
A = spzeros(2*H,2*H);
AT = spzeros(2*H,2*H);
g_r, V_r, a_dot = [zeros(H,2,1000) for i in 1:3]
KS, KD, S = [zeros(I_r) for i in 1:3]
ssb= 0
ssf =0
it_val = 0

for ir in 1:I_r

	r_r[ir]=r
	r_minr[ir] = r_min
	r_maxr[ir] = r_max

	KD[ir] = (α*A_prod/(r+δ))^(1/(1-α))*z_avg
	w = (1-α)*(A_prod*KD[ir].^(α))*z_avg^(-α)
    println(KD[ir])

	if ir>1
		v_0=V_r[:,:,ir-1]
	end

	v=v_0

	for n in 1:maxit
		V=v
		push!(V_n, V)

		#Forward difference
		dVf[1:H-1,:] = (V[2:H,:]-V[1:H-1,:])/da
		dVf[H,:] = (w*z + r.*a_max).^(-γ) # BC a<a_max

		#Backward difference
		dVb[2:H,:] = (V[2:H,:]-V[1:H-1,:])/da
		dVb[1,:] = (w*z + r.*a_min).^(-γ) # BC a>a_min

		# Consumption and savings with forward diff
		cf = dVf.^(-1/γ)
		ssf = w*zz + r.*aa - cf

		# Consumption and savings with backward diff
		cb = dVb.^(-1/γ)
		ssb = w*zz + r.*aa - cb

		# Consumption and Value function
		c0 = w*zz + r.*aa
		dV0 = c0.^(-γ)

		# Implement the upwind scheme, first choice back or Forward
		If = ssf .> 0
		Ib = ssb .< 0
		I0 = (1-If-Ib)

		dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0
		c = dV_Upwind.^(-1/γ)
		u = (c.^(1-γ))/(1-γ)
		# Now to construct the transition matrix

		# Now to constuct the A matrix
	    X =  -min(ssb, 0)/da
	    Y = -max(ssf, 0)/da + min(ssb, 0)/da
	    Z = max(ssf, 0)/da

	    A1 = spdiagm(Y[:,1]) + spdiagm(X[2:H,1],-1,H,H)+spdiagm([0;Z[1:H-1,1]])
		A2 = spdiagm(Y[:,2]) + spdiagm(X[2:H,2],-1,H,H)+spdiagm([0;Z[1:H-1,2]])

		A = [A1 spzeros(H,H); spzeros(H,H) A2] + A_switch
		B = (1/Δ +ρ)*speye(2*H) - A


		u_stacked = [u[:,1]; u[:,2]]
		V_stacked = [V[:,1];V[:,2]]

        b = u_stacked + V_stacked/Δ
        V_stacked = B\b

		V = [V_stacked[1:H] V_stacked[H+1:2*H]]
		V_change = V-v

		v=V

		push!(dist, findmax(abs(V_change))[1])
		if dist[n].< ε
			println("Value Function Converged Iteration=")
			println(n)
			break
		end
        println(n)
	end


#=========Now to find the SS for the KFE ==========#

	AT = A' # since the KFE transition matrix is just the transpose of HJBs
	b= zeros(2*H,1)

	# Fix a single value to prevent singularity

	i_fix =1
	b[i_fix]=.1
	for j=1:2*H
		AT[i_fix,j]=0;
	end
	AT[i_fix,i_fix] = 1.0

	# Now solve the linear system
	gg = AT\b[:,1]
	gg_sum = gg'*ones(2*H,1)*da
	gg = gg./gg_sum

	g = [gg[1:H] gg[H+1:2*H]]

	check_1 = g[:,1]'*ones(H,1).*da
	check_2 = g[:,2]'*ones(H,1).*da

	g_r[:,:,ir] = g
	a_dot[:,:,ir] = w*zz+r.*aa -c
	V_r[:,:,ir] = V

	KS[ir] = g[:,1]'*a.*da + g[:,2]'*a.*da
	S[ir] = KS[ir] - KD[ir]

	# Now update the interest rate

		if S[ir] > crit_S
			println("Excess Supply")
			r_max = r
			r= 0.5*(r+r_min)
		elseif S[ir] < -(crit_S)
			println("Excess Demand")
			r_min = r
			r= 0.5*(r+r_max)
		elseif abs(S[ir]<crit_S)
			println("Equilibrium found, interest rate=")
			println(r)
		break
	end
it_val = ir
end

V_r = V_r[:,:,1:40]
g_r = g_r[:,:,1:40]
