def _reduce_variance_interval(x, axis=None, biased=True, keepdims=False):
    # ddof=0 is biased variance (N), ddof=1 is unbiased variance (N-1)
    ddof = 0 if biased else 1
    return jnp.var(x, axis=axis, ddof=ddof, keepdims=keepdims)

def kernel_setup(warmup_length,
                 num_total_chains, num_super_chains,
                 naive,initialize_fn, randomKey,
                 target_log_prob_fn,init_step_size):
    key, init_key = random.split(randomKey)
    
    if naive:
      initial_position = initialize_fn((num_total_chains,), init_key)
    else:
        num_sub_chains = num_total_chains//num_super_chains
        initial_position_super = initialize_fn((num_super_chains,), init_key)
        initial_position = jnp.repeat(initial_position_super,num_sub_chains,axis=0)

    warmup = blackjax.chees_adaptation(
        target_log_prob_fn,num_chains=num_total_chains,
        target_acceptance_rate=0.75)
    optimizer = optax.adam(learning_rate=0.001)
    key_warmup, key_sample = random.split(key)
    (last_states, parameters), _= warmup.run(
        key_warmup,
        initial_position,
        init_step_size,
        optimizer,
        warmup_length
        )
    sample_keys = random.split(key_sample, num_total_chains)
    kernel = blackjax.dhmc(target_log_prob_fn, **parameters).step
    return kernel, sample_keys, last_states

def nested_rhat_constrained(result_state, num_super_chains,idx):
     # since we use only N=1, W_k is reduced to 0
    num_sub_chains = result_state.shape[0] // num_super_chains
    num_dimensions = result_state.shape[1]

    chain_states = result_state.reshape(1, -1, num_sub_chains, num_dimensions)
    # chain_states.shape = (1,16,128,2), so it is (N,K,M,D).
    # f_bar 1*k, mean_subchain.shape =(1,16,2), (N,K,D)
    mean_subchain = jnp.mean(chain_states, axis=2)

    variance_chain = _reduce_variance_interval(chain_states, axis=2, biased=False)
    # print(variance_chain.shape) # (1,16,2)
    W = jnp.mean(variance_chain, axis=1)
    # print(f"W dim: {W.shape}") # (1,2)
    B = _reduce_variance_interval(mean_subchain, axis=1, biased=False) # variance of between super chain

    r_hat = jnp.sqrt(1+B/W)[:,idx]
    return r_hat

def mse_calculation(result, mean_benchmark,
                    var_benchmark, mse_list):
    mc_mean = result.mean(axis=0)
    squared_error = (mc_mean - mean_benchmark)**2
    factor = 1/var_benchmark
    factored_sq_err = factor*squared_error
    mse = (factored_sq_err).mean()
    mse_list.append(mse)
    return mse_list, factored_sq_err

def simulation(warmup_length,num_total_chains, num_super_chains,
               naive,initialize_fn, randomKeys,
               target_log_prob_fn,init_step_size,
               true_mean,true_var):
    result_mse = []
    kernel, sample_keys, last_states = kernel_setup(warmup_length,num_total_chains, 
                                                      num_super_chains, naive,initialize_fn,
                                                      randomKeys,target_log_prob_fn,init_step_size)
    sample_states, info = jax.vmap(kernel)(sample_keys, last_states)
    samples = sample_states.position
    dims = samples.shape[1]
    result_mse, factored_sq_err = mse_calculation(samples,true_mean,true_var,result_mse)
    rhat_list = []
    for dim in range(dims):
      rhat = nested_rhat_constrained(samples, num_super_chains, dim)[-1]
      rhat_list.append(rhat)
    mean_rhat = jnp.mean(jnp.stack(rhat_list))
    del kernel, sample_keys, last_states, sample_states, info, result_mse,rhat_list
    gc.collect()
    jax.clear_caches()
    return samples, mean_rhat

def single_chain(warmup_length, sample_length,
                 initialize_fn, randomKey,
                 target_log_prob_fn, init_step_size):
    initial_position = initialize_fn((1,), randomKey)

    warmup = blackjax.chees_adaptation(
        target_log_prob_fn,
        num_chains=1,
        target_acceptance_rate=0.75,
    )

    optimizer = optax.adam(learning_rate=0.001)
    key_warmup, key_sample = jax.random.split(randomKey)
    (last_states, parameters), _ = warmup.run(
        key_warmup,
        initial_position,
        init_step_size,
        optimizer,
        warmup_length,
    )

    BJX_dhmc = blackjax.dhmc(target_log_prob_fn,**parameters)

    initial_state = jax.tree.map(
        lambda x: x[0],
        last_states,
    )

    kernel = jax.jit(BJX_dhmc.step)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    states = inference_loop(key_sample, kernel, initial_state, sample_length)

    mcmc_samples = states.position
    sample_mean = mcmc_samples.mean(axis=0)
    in_chain_var = _reduce_variance_interval(mcmc_samples, axis=0, biased=False)

    return mcmc_samples, sample_mean, in_chain_var


def multiple_chains(warmup_length, sample_length,initialize_fn, randomKey,
                    target_log_prob_fn, init_step_size, sample_list, repitition):
    keys = jax.random.split(randomKey, repitition)
    if repitition > 1:
      calculate = True
      in_chain_var_list = []
      in_chain_mean_list = []
    else:
      calculate = False

    for iteration in range(repitition):
        key = keys[iteration]
        samples, mean, var = single_chain(warmup_length, sample_length,
                               initialize_fn, key,
                               target_log_prob_fn, init_step_size)
        # check number of repition, and run R-hat calculation.
        if calculate:
          in_chain_var_list.append(var)
          in_chain_mean_list.append(mean)

        sample_list.append({
            "Iteration": iteration,
            "Samples": samples,
        })
    if calculate:
      in_chain_mean_array = jnp.stack(in_chain_mean_list)
      in_chain_var_array = jnp.stack(in_chain_var_list)
      W = jnp.mean(in_chain_var_array)
      B = sample_length*_reduce_variance_interval(in_chain_mean_array, axis=0, biased=False)
      r_hat = jnp.sqrt((sample_length-1)/sample_length + (B/W)*(1/sample_length))
      mean_r_hat = r_hat.mean()
    else :
      mean_r_hat = None
    return sample_list, mean_r_hat

#===============================
# Plotting function:
#===============================
# MSC stands for Many-Short_Chain
def comparison_plot(samples, multichain_samples, 
                    MSC_C, MSC_N, multichain_r, 
                    MSC_C_r, MSC_N_r):

    fig, axes = plt.subplots(2, 2, figsize=(16, 12),dpi=150)
    axes = axes.flatten()
    # Single Chain
    one_chain = samples[0]["Samples"]
    axes[0].plot(one_chain[:, 0], one_chain[:, 1],
                 marker="o", markersize=2, linewidth=0.5,alpha=0.7)

    axes[0].set_title(f"One Chain, Sample Length = {one_chain.shape[0]}")
    axes[0].set_xlabel(r"$\theta_1$")
    axes[0].set_ylabel(r"$\theta_2$")

    # Consecutive Chains

    for chain in multichain_samples:
       chain_samples = chain["Samples"]
       iteration = chain["Iteration"]
       axes[1].plot(chain_samples[:, 0], chain_samples[:, 1],
                    marker="o", markersize=2, linewidth=0.5,
                    alpha=0.7, label=f"Chain {iteration}")
    axes[1].set_title(f"Multiple Chains, rhat={float(multichain_r):.4f}, each sample length = {chain_samples.shape[0]}")

    axes[1].set_xlabel(r"$\theta_1$")
    axes[1].set_ylabel(r"$\theta_2$")
    axes[1].legend()

    # Many-Short_Chain, Constrained Initialization

    axes[2].scatter(MSC_C[:, 0], MSC_C[:, 1],
                    s=5, alpha=0.5)

    axes[2].set_title(
        f"Many-Short-Chain, constrained, K = {num_super_chains}, M={M}, "
        f"nested rhat={float(MSC_C_r):.4f}")

    axes[2].set_xlabel(r"$\theta_1$")
    axes[2].set_ylabel(r"$\theta_2$")

    # Many-Short-Chain, naive Initialization
    axes[3].scatter( MSC_N[:, 0], MSC_N[:, 1],
                    s=5, alpha=0.5)

    axes[3].set_title( f"Many-Short-Chain, naive, K = {num_chains_short}, M=1, "
                      f"nested rhat={float(MSC_N_r):.4f}")
    axes[3].set_xlabel(r"$\theta_1$")
    axes[3].set_ylabel(r"$\theta_2$")

    all_x = []
    all_y = []

    # One chain
    all_x.append(one_chain[:, 0])
    all_y.append(one_chain[:, 1])

    # Multiple chains
    for chain in multichain_samples:
        all_x.append(chain["Samples"][:, 0])
        all_y.append(chain["Samples"][:, 1])

    # Other samples
    all_x.append(MSC_C[:, 0])
    all_y.append(MSC_C[:, 1])

    all_x.append(MSC_N[:, 0])
    all_y.append(MSC_N[:, 1])

    x_min = min(x.min() for x in all_x)
    x_max = max(x.max() for x in all_x)

    y_min = min(y.min() for y in all_y)
    y_max = max(y.max() for y in all_y)

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.show()