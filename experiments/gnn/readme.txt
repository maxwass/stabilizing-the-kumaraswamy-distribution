
train_edge_latent.py: responsible for running the VEE model architectures with varying latent distributions

train_gae.py: gae = graph auto encoders. No latent distribution. Just regular auto-encoder.

run_training.py: this trains all models as displayed in the paper. The entries in list `commands` which are not commented will be executed.

To create figure plots, use gnn_plot.ipynb