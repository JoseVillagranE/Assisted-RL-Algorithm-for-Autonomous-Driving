from .DummyModel import DummyModel
from .manual_model import Manual_Model
from .DDPG import DDPG
from .BC import BC
from .CoL import CoL

def init_model(model_name,
               model_type,
               state_dim,
               action_space,
               h_image_in,
               w_image_in,
               z_dim,
               actor_lr,
               critic_lr,
               batch_size,
               optim,
               gamma,
               tau,
               alpha,
               beta,
               type_RM,
               max_memory_size,
               device ='cpu',
               rw_weights=None,
               actor_linear_layers=[],
               pretraining_steps=100,
               lambdas=[1,1,1],
               expert_prop=0.25,
               agent_prop=0.75,
               rm_filename="BC-1.npy",
               ou_noise_mu=0.0,
               ou_noise_theta=0.6,
               ou_noise_max_sigma=0.4,
               ou_noise_min_sigma=0.0,
               ou_noise_decay_period=250,
               wp_encode=False,
               wp_encoder_size=64):

    model = None
    
    if model_name == "DummyModel":
        model = DummyModel(action_space)

    elif model_name == "manual_model":
        model = Manual_Model(2)

    elif model_name == "DDPG":
        model = DDPG(state_dim,
                    action_space,
                    h_image_in,
                    w_image_in,
                    actor_lr = actor_lr,
                    critic_lr = critic_lr,
                    optim=optim,
                    batch_size = batch_size,
                    gamma = gamma,
                    tau = tau,
                    alpha = alpha,
                    beta = beta,
                    model_type=model_type,
                    z_dim=z_dim,
                    type_RM = type_RM,
                    max_memory_size = max_memory_size,
                    device=device,
                    rw_weights=rw_weights,
                    actor_linear_layers=actor_linear_layers)
        
    elif model_name == "BC": # for the moment is just for evaluation
        model = BC(state_dim=state_dim,
                   action_space=action_space,
                   type_AC=model_type,
                   VAE_weights_path="./models/weights/segmodel_expert_samples_sem_369.pt")
        
    elif model_name=="CoL":
        model = CoL(pretraining_steps=pretraining_steps,
                    state_dim=state_dim,
                    action_space=action_space,
                    batch_size=batch_size,
                    expert_prop=expert_prop,
                    agent_prop=agent_prop,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    gamma=gamma,
                    tau=tau,
                    optim=optim,
                    rw_weights=rw_weights,
                    lambdas=lambdas,
                    model_type=model_type,
                    z_dim=z_dim,
                    beta=beta,
                    type_RM=type_RM,
                    max_memory_size=max_memory_size,
                    rm_filename=rm_filename,
                    ou_noise_mu=ou_noise_mu,
                    ou_noise_theta=ou_noise_theta,
                    ou_noise_max_sigma=ou_noise_max_sigma,
                    ou_noise_min_sigma=ou_noise_min_sigma,
                    ou_noise_decay_period=ou_noise_decay_period,
                    wp_encode=wp_encode,
                    wp_encoder_size=wp_encoder_size)
    else:
        raise NotImplementedError("Dont exist that model that you required")
    return model
