def bivariate_MVN_sampler(var1, var2, rho, num_dim, repitition,
                          warmup_length, num_chains_short, num_super_chains,save_dir):
  # mean array
  mu = jnp.full(num_dim, 0.0)
  # covariance matrix
  cov = jnp.array([
    [var1, rho*jnp.sqrt(var1*var2)],
    [rho*jnp.sqrt(var1*var2), var2]])
  target = tfd.MultivariateNormalFullCovariance(
      loc=jnp.array(mu),
      covariance_matrix=cov)
  init_step_size = 0.5
  def target_log_prob_fn(x):
    return target.log_prob(x)
  def initialize(shape, key):
    offset = 2
    return 10*random.normal(key, shape + (num_dim,)) + offset
  mean_benchmark = target.mean()
  var_benchmark = target.variance()

  #simulation part:
  Aniso_MSE_c_list = []
  Aniso_MSE_n_list = []
  Aniso_RHat_c_list = []
  Aniso_RHat_n_list = []

  base_key = random.PRNGKey(0)
  keys = random.split(base_key, repitition)
  for length in warmup_length:
    mem(f"Simulation Start")
    simulation(length,num_chains_short, num_super_chains,
               False,initialize, keys, target_log_prob_fn,init_step_size,
               repitition, Aniso_RHat_c_list,Aniso_MSE_c_list,mean_benchmark,var_benchmark)
    simulation(length,num_chains_short, num_super_chains,
               True,initialize, keys, target_log_prob_fn,init_step_size,
               repitition, Aniso_RHat_n_list,Aniso_MSE_n_list, mean_benchmark,var_benchmark)
  # save to dataframe and files
  MSE_c_df = pd.DataFrame(Aniso_MSE_c_list)
  R_Hat_c_df = pd.DataFrame(Aniso_RHat_c_list)
  MSE_n_df = pd.DataFrame(Aniso_MSE_n_list)
  R_Hat_n_df = pd.DataFrame(Aniso_RHat_n_list)

  # Use a descriptive parameter label
  label = f"var1_{var1:g}_var2_{var2:g}_rho_{rho:g}"

  MSE_c_df.to_pickle(f"{save_dir}/{label}_MSE_c.pkl")
  MSE_n_df.to_pickle(f"{save_dir}/{label}_MSE_n.pkl")
  R_Hat_c_df.to_pickle(f"{save_dir}/{label}_Rhat_c.pkl")
  R_Hat_n_df.to_pickle(f"{save_dir}/{label}_Rhat_n.pkl")

  del MSE_c_df, R_Hat_c_df, MSE_n_df, R_Hat_n_df
  gc.collect()
  print("Simulation Done, Data Saved!")
