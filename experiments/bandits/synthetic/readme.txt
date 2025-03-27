generate_bandit_data.ipynb: generates synthetic data for the multi-armed bandit experiment presented in the paper, and construct plot showing effect of exponentiaiton on probabilities.

vbe_config.yml: houses parameters common to vbe models.

vbe_train.py: responsible for running the VEE model architectures with varying latent distributions.

lmcts_config.yml: houses parameters for lmcts model.

lmcts_train.py: responsible for running the langevin monte carlo - thompson sampling baseline model.

run_training.py: this trains all models as displayed in the paper. The entries in list `commands` which are not commented will be executed.

To create figure plots, use viz_bandits.ipynb.