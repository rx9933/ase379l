from matplotlib import pyplot as plt
from mcmc import *
def dram_combined(q0, s2_0, J_func, D, r_calc, M, ns, sigma2_s, n_obs, rod, x_obs, T_obs, 
                  rng_seed=None, k0=100, gamma2=1/25):
    """
    DRAM: Delayed Rejection + Adaptive Metropolis + Gibbs sampling
    
    Parameters:
    q0: initial q sample [Phi, h]
    s2_0: initial sigma_squared
    J_func: proposal function that returns (proposal, covariance)
    D: initial proposal covariance
    r_calc: function to compute acceptance ratio (uses current sigma2)
    M: number of samples
    ns, sigma2_s: hyperparameters for inverse gamma prior
    n_obs: number of observations
    rod, x_obs, T_obs: problem data
    rng_seed: random seed
    k0: when to start adaptation (default 100)
    gamma2: scaling for delayed rejection (default 1/25)
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    p = len(q0)  # dimension (should be 2)
    q_current = np.array(q0, dtype=float)
    s2_current = s2_0
    
    # Store samples (q + sigma2)
    samples = np.zeros((M, p + 1))
    samples[0, :p] = q_current
    samples[0, p] = s2_current
    
    # AM parameters
    sp = 2.38**2 / p  # scaling factor
    
    # Storage for adaptation
    Vk = D.copy()  # current covariance for adaptation
    proposal_cov = D.copy()  # current proposal covariance
    
    accepted = 0
    Vk_at_101 = None
    
    print("Running DRAM (Delayed Rejection + Adaptive Metropolis + Gibbs)...")
    
    for i in range(1, M):
        # ============================================================
        # STEP 1: ADAPTIVE PROPOSAL FOR q
        # ============================================================
        
        if i < k0:
            # First k0 iterations: use fixed proposal from J_func
            q_proposed, prop_cov = J_func(q_current, returnd=True)
            if prop_cov is None:
                prop_cov = D
            proposal_cov = prop_cov
        else:
            # Adaptive Metropolis proposal
            if i == k0:
                # First time using AM (k=101)
                samples_so_far = samples[:k0, :p]
                Vk = np.cov(samples_so_far, rowvar=False)
                # Add small diagonal for numerical stability
                Vk += 1e-8 * np.eye(p)
                Vk_at_101 = Vk.copy()
                print(f"\nAt k=101, computed Vk from first {k0} samples")
            
            elif np.mod(i - k0, k0) == 0 and i > k0:
                # Update covariance periodically
                samples_so_far = samples[:i, :p]
                Vk = np.cov(samples_so_far, rowvar=False)
                Vk += 1e-8 * np.eye(p)  # Regularization
            
            # Scale covariance for optimal acceptance
            proposal_cov = sp * Vk
            
            # Generate proposal
            q_proposed = np.random.multivariate_normal(q_current, proposal_cov)
        
        # ============================================================
        # STEP 2: FIRST STAGE METROPOLIS ACCEPTANCE
        # ============================================================
        # Compute acceptance ratio using CURRENT sigma_squared
        r1 = r_calc(q_proposed, q_current, s2_current)
        
        # Handle numerical issues
        if not np.isfinite(r1) or r1 < 0:
            r1 = 0
        alpha1 = min(1, r1)
        
        stage1_accepted = False
        
        if np.random.rand() < alpha1:
            # Accept first stage proposal
            q_current = q_proposed
            accepted += 1
            stage1_accepted = True
        
        # ============================================================
        # STEP 3: DELAYED REJECTION (if first stage rejected)
        # ============================================================
        if not stage1_accepted:
            try:
                # Second stage proposal with reduced covariance
                proposal_cov2 = gamma2**2 * proposal_cov
                # Ensure symmetry and positive definiteness
                proposal_cov2 = (proposal_cov2 + proposal_cov2.T) / 2
                proposal_cov2 += 1e-10 * np.eye(p)
                
                q_proposed2 = np.random.multivariate_normal(q_current, proposal_cov2)
                
                # Compute ratios for second stage
                r2 = r_calc(q_proposed2, q_current, s2_current)
                if not np.isfinite(r2) or r2 < 0:
                    r2 = 0
                
                # Need alpha1 for reverse move (q_proposed -> q_proposed2)
                r1_reverse = r_calc(q_proposed, q_proposed2, s2_current)
                if not np.isfinite(r1_reverse) or r1_reverse < 0:
                    r1_reverse = 0
                alpha1_reverse = min(1, r1_reverse)
                
                # Compute proposal ratio J(q*|q*2)/J(q*|q_{k-1})
                # For multivariate normal proposals
                try:
                    # Use Cholesky for numerical stability
                    L = np.linalg.cholesky(proposal_cov)
                    
                    diff1 = q_proposed - q_proposed2
                    diff2 = q_proposed - q_current
                    
                    # Solve L * x = diff
                    x1 = np.linalg.solve(L, diff1)
                    x2 = np.linalg.solve(L, diff2)
                    
                    log_J_ratio = -0.5 * (np.sum(x1**2) - np.sum(x2**2))
                    log_J_ratio = np.clip(log_J_ratio, -500, 500)
                    J_ratio = np.exp(log_J_ratio)
                except:
                    # Fallback if Cholesky fails
                    J_ratio = 1.0
                
                # Compute alpha2 according to theory
                numerator = r2 * J_ratio * (1 - alpha1_reverse)
                denominator = max(1 - alpha1, 1e-12)
                
                alpha2 = numerator / denominator
                alpha2 = min(1, max(0, alpha2))  # Clamp to [0,1]
                
                if np.isfinite(alpha2) and np.random.rand() < alpha2:
                    q_current = q_proposed2
                    accepted += 1
                    
            except Exception as e:
                # If delayed rejection fails, keep current state
                pass
        
        # ============================================================
        # STEP 4: GIBBS UPDATE FOR sigma_squared
        # ============================================================
        # Compute sum of squared errors for current q
        Phi, h = q_current
        rod.Phi_val = Phi
        rod.h_val = h
        
        try:
            T_model = rod.Ts(x_obs)
            if np.all(np.isfinite(T_model)):
                SSq = np.sum((T_obs - T_model)**2)
                # Clip to prevent extreme values
                SSq = np.clip(SSq, 1e-10, 1e10)
            else:
                SSq = 1.0  # Default if model fails
        except:
            SSq = 1.0
        
        # Inverse Gamma parameters
        aval = ns/2 + n_obs/2
        bval = (ns * sigma2_s)/2 + SSq/2
        bval = max(bval, 1e-10)  # Ensure positive
        
        # Sample from Inv-Gamma using relationship with Gamma
        gamma_sample = np.random.gamma(aval, 1/bval)
        if gamma_sample > 0 and np.isfinite(gamma_sample):
            s2_current = 1 / gamma_sample
        else:
            s2_current = s2_0  # Fallback to initial value
        
        # ============================================================
        # STEP 5: STORE SAMPLES
        # ============================================================
        samples[i, :p] = q_current
        samples[i, p] = s2_current
        
        # Progress reporting
        if (i + 1) % 1000 == 0:
            recent_acc = accepted / (i + 1)
            print(f"  Iteration {i + 1}/{M}, acceptance rate: {recent_acc:.3f}, σ² = {s2_current:.4f}")
    
    acceptance_rate = accepted / (M - 1)
    print(f"\nFinal acceptance rate: {acceptance_rate:.3f}")
    
    return samples, acceptance_rate, Vk_at_101, Vk


# Create a wrapper function that matches your existing interface
def dram_wrapper(q0, s2_0, J_func, D, r_calc, M, ns, sigma2_s, n_obs, rod, x_obs, T_obs, 
                 rng_seed=None, k=None):
    """
    Wrapper for DRAM that matches the interface expected by run_adaptive_metropolis_for_heat_equation
    """
    k0 = k if k is not None else 100
    return dram_combined(q0, s2_0, J_func, D, r_calc, M, ns, sigma2_s, n_obs, rod, x_obs, T_obs,
                         rng_seed=rng_seed, k0=k0)


# Also need to modify r_calc to accept sigma2 parameter
def create_dram_posterior_ratio_func(rod, x_obs, T_obs):
    """
    Create posterior ratio function for DRAM that accepts sigma2 as a parameter
    """
    # Prior parameters (from your code)
    prior_Phi_mean = -15
    prior_Phi_std = 10
    prior_h_mean = 0.001
    prior_h_std = 0.005
    
    def dram_r_calc(q_star, q_current, s2):
        """
        Compute π(q_star|data,s2) / π(q_current|data,s2)
        This is the ratio needed for Metropolis step with current sigma2
        """
        try:
            # Check for physical constraints
            if q_star[1] <= 0 or q_current[1] <= 0:
                return 0
            
            # Compute model predictions for both q's
            rod.Phi_val, rod.h_val = q_star
            T_star = rod.Ts(x_obs)
            
            rod.Phi_val, rod.h_val = q_current
            T_current = rod.Ts(x_obs)
            
            # Check for valid outputs
            if not (np.all(np.isfinite(T_star)) and np.all(np.isfinite(T_current))):
                return 0
            
            # Log-likelihoods (using current sigma2)
            n = len(x_obs)
            log_lik_star = -0.5 * np.sum((T_obs - T_star)**2 / s2)
            log_lik_current = -0.5 * np.sum((T_obs - T_current)**2 / s2)
            
            # Log-priors
            log_prior_star = (norm.logpdf(q_star[0], prior_Phi_mean, prior_Phi_std) +
                              norm.logpdf(q_star[1], prior_h_mean, prior_h_std))
            log_prior_current = (norm.logpdf(q_current[0], prior_Phi_mean, prior_Phi_std) +
                                 norm.logpdf(q_current[1], prior_h_mean, prior_h_std))
            
            # Log ratio (proposal symmetric, so cancels)
            log_ratio = (log_lik_star + log_prior_star) - (log_lik_current + log_prior_current)
            
            # Clip to prevent overflow
            log_ratio = np.clip(log_ratio, -500, 500)
            
            return np.exp(log_ratio)
            
        except Exception as e:
            return 0
    
    return dram_r_calc


# Modified run function to use DRAM properly
def run_dram_for_heat_equation(M=10000, k_for_am=100, burn=3000, title='DRAM', data_filename='HW06_Problem3.mat'):
    """
    Run DRAM algorithm for the heat equation problem with unknown sigma_squared
    """
    # Setup problem
    rod, x_obs, T_obs, sigma2_0, q0 = setup_heat_equation_problem(data_filename)
    
    # Initial proposal covariance
    D = np.array([[1e-2, 0],
                  [0, 2e-10]])
    
    # Create initial proposal function
    def initial_proposal_func(q_current, returnd=False):
        return (
            np.random.multivariate_normal(q_current, D),
            D
        ) if returnd else np.random.multivariate_normal(q_current, D)
    
    # Create DRAM-specific posterior ratio function
    dram_r_calc = create_dram_posterior_ratio_func(rod, x_obs, T_obs)
    
    # Hyperparameters for inverse gamma prior
    ns = 0.01
    sigma2_s = 4.0
    
    print("\n" + "="*70)
    print("RUNNING DRAM: Delayed Rejection + Adaptive Metropolis + Gibbs")
    print("="*70)
    print(f"M = {M} iterations, burn-in = {burn}")
    print(f"k0 = {k_for_am} (start adaptation after {k_for_am} samples)")
    print(f"γ² = 1/25 (delayed rejection scaling)")
    print(f"sp = 2.38²/{len(q0)} = {2.38**2/len(q0):.4f}")
    print(f"ns = {ns}, σ²_s = {sigma2_s}")
    
    # Run DRAM
    samples, acceptance_rate, Vk_at_101, Vk_at_end = dram_wrapper(
        q0=q0, s2_0=sigma2_0,
        J_func=initial_proposal_func, D=D,
        r_calc=dram_r_calc,
        M=M, ns=ns, sigma2_s=sigma2_s,
        n_obs=len(x_obs), rod=rod,
        x_obs=x_obs, T_obs=T_obs,
        rng_seed=42, k=k_for_am
    )
    
    print(f"\nFinal acceptance rate: {acceptance_rate:.3f}")
    print(f"\nVk at k=101:\n{Vk_at_101}")
    print(f"\nFinal Vk:\n{Vk_at_end}")
    
    # Apply burn-in
    samples_post_burn = samples[burn:]
    q_samples_post = samples_post_burn[:, :-1]
    s2_samples_post = samples_post_burn[:, -1]
    
    # Compute posterior values for MAP estimate
    print("\nComputing posterior values for MAP estimate...")
    
    prior_Phi_mean = -15
    prior_Phi_std = 10
    prior_h_mean = 0.001
    prior_h_std = 0.005
    
    posterior_values = np.zeros(len(q_samples_post))
    for i, (q, s2) in enumerate(zip(q_samples_post, s2_samples_post)):
        Phi, h = q
        rod.Phi_val = Phi
        rod.h_val = h
        T_model = rod.Ts(x_obs)
        
        # Likelihood with current sigma_squared
        n = len(x_obs)
        # Use log domain for numerical stability
        log_likelihood = -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * np.sum((T_obs - T_model)**2 / s2)
        
        # Log priors
        log_prior_q = (norm.logpdf(Phi, prior_Phi_mean, prior_Phi_std) +
                       norm.logpdf(h, prior_h_mean, prior_h_std))
        
        # Log prior for sigma_squared (Inverse Gamma)
        log_prior_sigma = -(ns/2 + 1) * np.log(s2) - (ns * sigma2_s) / (2 * s2)
        
        # Log posterior
        log_posterior = log_likelihood + log_prior_q + log_prior_sigma
        
        # Convert to actual posterior (clipping for stability)
        log_posterior = np.clip(log_posterior, -700, 700)
        posterior_values[i] = np.exp(log_posterior)
    
    # Find MAP estimate
    map_idx = np.argmax(posterior_values)
    q_map = q_samples_post[map_idx]
    s2_map = s2_samples_post[map_idx]
    post_map = posterior_values[map_idx]
    
    # Print results
    print(f"\n{'='*60}")
    print("DRAM RESULTS")
    print(f"{'='*60}")
    print(f"\nMAP Estimate:")
    print(f"  Φ = {q_map[0]:.6f} W/cm²")
    print(f"  h = {q_map[1]:.6f} W/cm²/°C")
    print(f"  σ² = {s2_map:.6f} (°C)²")
    print(f"  σ = {np.sqrt(s2_map):.4f} °C")
    print(f"  π(data|q_MAP,σ²_MAP)π0(q_MAP,σ²_MAP) = {post_map:.2e}")
    
    print(f"\nPosterior Summary (after {burn} burn-in):")
    print(f"{'='*60}")
    print(f"Φ:")
    print(f"  Mean = {np.mean(q_samples_post[:, 0]):.6f} W/cm²")
    print(f"  Std  = {np.std(q_samples_post[:, 0]):.6f} W/cm²")
    print(f"  95% CI = [{np.percentile(q_samples_post[:, 0], 2.5):.6f}, "
          f"{np.percentile(q_samples_post[:, 0], 97.5):.6f}]")
    
    print(f"\nh:")
    print(f"  Mean = {np.mean(q_samples_post[:, 1]):.6e} W/cm²/°C")
    print(f"  Std  = {np.std(q_samples_post[:, 1]):.6e} W/cm²/°C")
    print(f"  95% CI = [{np.percentile(q_samples_post[:, 1], 2.5):.6e}, "
          f"{np.percentile(q_samples_post[:, 1], 97.5):.6e}]")
    
    print(f"\nσ²:")
    print(f"  Mean = {np.mean(s2_samples_post):.6f} (°C)²")
    print(f"  Std  = {np.std(s2_samples_post):.6f} (°C)²")
    print(f"  95% CI = [{np.percentile(s2_samples_post, 2.5):.6f}, "
          f"{np.percentile(s2_samples_post, 97.5):.6f}]")
    
    # Create diagnostic plots
    create_dram_diagnostic_plots(samples_post_burn, q_map, s2_map, x_obs, T_obs, rod, 
                                 posterior_values, acceptance_rate, title)
    
    return samples, posterior_values, q_map, Vk_at_101, acceptance_rate


def create_dram_diagnostic_plots(samples, q_map, s2_map, x_obs, T_obs, rod, 
                                 posterior_values, acceptance_rate, title):
    """Create diagnostic plots for DRAM results"""
    q_samples = samples[:, :-1]
    s2_samples = samples[:, -1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Phi trace
    axes[0, 0].plot(q_samples[:, 0], alpha=0.7, color='skyblue', linewidth=0.5)
    axes[0, 0].axhline(y=q_map[0], color='red', linestyle='--', label=f'MAP: {q_map[0]:.4f}')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Φ (W/cm²)')
    axes[0, 0].set_title('Trace Plot: Φ')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: h trace
    axes[0, 1].plot(q_samples[:, 1], alpha=0.7, color='lightcoral', linewidth=0.5)
    axes[0, 1].axhline(y=q_map[1], color='red', linestyle='--', label=f'MAP: {q_map[1]:.6f}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('h (W/cm²/°C)')
    axes[0, 1].set_title('Trace Plot: h')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: sigma_squared trace
    axes[0, 2].plot(s2_samples, alpha=0.7, color='lightgreen', linewidth=0.5)
    axes[0, 2].axhline(y=s2_map, color='red', linestyle='--', label=f'MAP: {s2_map:.4f}')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('σ² (°C)²')
    axes[0, 2].set_title('Trace Plot: σ²')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Phi histogram
    axes[1, 0].hist(q_samples[:, 0], bins=30, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black')
    axes[1, 0].axvline(q_map[0], color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Φ (W/cm²)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Marginal Posterior π(Φ|data)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: h histogram
    axes[1, 1].hist(q_samples[:, 1], bins=30, density=True, alpha=0.7, 
                    color='lightcoral', edgecolor='black')
    axes[1, 1].axvline(q_map[1], color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('h (W/cm²/°C)')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Marginal Posterior π(h|data)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Model fit
    rod.Phi_val, rod.h_val = q_map
    T_map = rod.Ts(x_obs)
    
    axes[1, 2].plot(x_obs, T_obs, 'ko', markersize=8, label='Observations')
    axes[1, 2].plot(x_obs, T_map, 'r-', linewidth=2, label='MAP prediction')
    axes[1, 2].fill_between(x_obs, T_map - 2*np.sqrt(s2_map), T_map + 2*np.sqrt(s2_map),
                            color='red', alpha=0.2, label=f'±2σ (σ={np.sqrt(s2_map):.2f}°C)')
    axes[1, 2].set_xlabel('Position x (cm)')
    axes[1, 2].set_ylabel('Temperature (°C)')
    axes[1, 2].set_title('Model Fit with MAP Parameters')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} Results (Acceptance Rate: {acceptance_rate:.3f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{title}_diagnostics.png', dpi=150)
    plt.show()