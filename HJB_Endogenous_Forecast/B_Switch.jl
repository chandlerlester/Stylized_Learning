#==============================================================================

		B_Switch Code

==============================================================================#

#==============================================================================

    Now we are going to construct a matrix summarizing the evolution of V_z

    This comes from the following discretized Bellman equation:

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + v_z⋅μ(z)
                                   + 1/2(v_{zz})σ^2(z)

                                   or

    ρv_ij = u(c_ij) + v_k(zF(k_{i}) -δk-c) + ((v_{i,j+1}-v_{i,j})/Δz)μ(z)
                            + 1/2((v_{i,j+1}-2v_{i,j}+v_{ij-1})/Δz^2)σ^2(z)

    Assume forward difference because of boundary conditions

==============================================================================#

function B_switch(μ, Σ_sq, dz, H)
	 dz_sq = dz^2
	 yy = (-Σ_sq/dz_sq - μ/dz)
	 χ = Σ_sq/(2*dz_sq)
	 ζ = μ/dz + Σ_sq/(2*dz_sq)


	 # Define the Diagonals of this matrix
	 updiag_z = zeros(H,1)
	 	for j = 1:J
			updiag_z =[updiag_z; repeat([ζ[j]], H, 1)]
		end
	 updiag_z =(updiag_z[:])


	 centerdiag_z=repeat([χ[1]+yy[1]],H,1)
		for j = 2:J-1
			centerdiag_z = [centerdiag_z; repeat([yy[j]], H, 1)]
		end
	 centerdiag_z=[centerdiag_z; repeat([yy[J]+ζ[J]], H, 1)]
	  centerdiag_z = centerdiag_z[:]

	lowdiag_z = repeat([χ[2]], H, 1)
		for j=3:J
			lowdiag_z = [lowdiag_z; repeat([χ[j]],H,1)]
		end
	lowdiag_z=lowdiag_z[:]

	# spdiags in Matlab allows for automatic trimming/adding of zeros
	    # spdiagm does not do this

	B_switch = sparse(Diagonal(centerdiag_z))+ [zeros(H,H*J);  sparse(Diagonal(lowdiag_z)) zeros(H*(J-1), H)]+ sparse(Diagonal(updiag_z))[(H+1):end,1:(H*J)] # trim off rows of zeros

return B_switch

end
