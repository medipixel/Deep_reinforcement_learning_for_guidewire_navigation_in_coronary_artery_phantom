agent = dict(
    type="DQfDAgent",
    hyper_params=dict(
        gamma=0.99,
        tau=5e-3,
        buffer_size=int(1e5),  # Openai baselines: int(1e4)
        batch_size=32,  # Openai baselines: 32
        update_starts_from=int(2e4),  # Openai baselines: int(1e4)
        multiple_update=1,  # Multiple learning updates
        train_freq=4,  # In openai baselines, train_freq = 4
        gradient_clip=10.0,  # Dueling: 10.0
        n_step=10,
        w_n_step=1.0,
        w_q_reg=1e-7,
        per_alpha=0.4,  # Openai baselines: 0.6
        per_beta=0.6,
        per_eps=1e-3,
        # fD
        per_eps_demo=1.0,
        lambda1=1.0,  # N-step return weight
        lambda2=1.0,  # Supervised loss weight
        # lambda3 = weight_decay (l2 regularization weight)
        margin=0.8,
        pretrain_step=int(1e3),
        # Epsilon Greedy
        max_epsilon=1.0,  # Use epsilon greedy
        min_epsilon=0.01,  # Openai baselines: 0.01
        epsilon_decay=1.0,  # Openai baselines: 1e-7 / 1e-1
        # Use bifurcation dependent action
        use_bifur_action=False,
        # grad_cam
        grad_cam_layer_list=[
            "backbone.cnn.cnn_0.cnn",
            "backbone.cnn.cnn_1.cnn",
            "backbone.cnn.cnn_2.cnn",
        ],
    ),
    learner_cfg=dict(
        type="DQNLearner",
        # Distributional Loss
        loss_type=dict(type="C51Loss"),
        backbone=dict(
            type="CNN",
            configs=dict(
                input_sizes=[4, 32, 64],
                output_sizes=[32, 64, 64],
                kernel_sizes=[8, 4, 3],
                strides=[4, 2, 1],
                paddings=[1, 0, 0],
            ),
        ),
        head=dict(
            type="C51DuelingMLP",
            configs=dict(
                hidden_sizes=[512],
                v_min=-2.0,
                v_max=0.0,
                atom_size=51,
                output_activation="identity",
                # NoisyNet
                use_noisy_net=True,
                std_init=0.5,
            ),
        ),
        optim_cfg=dict(
            lr_dqn=1e-4,  # Dueling: 6.25e-5, openai baselines: 1e-4
            weight_decay=1e-5,  # This makes saturation in cnn weights
            adam_eps=1e-8,  # Rainbow: 1.5e-4, openai baselines: 1e-8
        ),
    ),
)