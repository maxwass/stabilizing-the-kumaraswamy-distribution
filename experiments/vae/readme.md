 To run the VAE experiments, run `python run_training.py`.
 
 To evaluate the model(s) for train & test metrics, run `python run_eval.py`. This will output a text file for each model in the `experiments/vae/results` directory.


 Misc Notes
 - The KS and Beta are defined on $(0, 1)$. For data with observations on $0$ or $1$, we project $0$ values and $1$ values to $.5/255$ and $1 - .5/255$ respectively.

- Beta Variational Posterior Instability
    -- We attempt to train with 2 link functions enforcing the positivity of the encoded Beta distributional parameters: softplus and exp. To see which is used in the results, view the corresponding file in vae/results/.txt and look for "var_link_func".
   -- If no model/results file exists, no stable training was found

- Cifar10-KS-Beta Instability
   -- Mostly trains just fine but did find a training run with instability. Using softplus Beta likelihood decoder. For exp Beta likelihood decoder, see the model trained_models/cifar10_ks_beta_exp_decoder.ckpt. Will need to manually alter VAE model code: softplus --> exp in decoder.

