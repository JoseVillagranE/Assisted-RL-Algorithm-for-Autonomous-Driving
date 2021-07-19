import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


class MDN_RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        seq_len,
        batch_size,
        device,
        num_layers=1,
        action_size=2,
        gaussians=3,
        mode="inference",
    ):
        """
        input_size -> z_dim + Compl
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.gaussians = gaussians
        self.mode = mode
        self.lstm = LSTM(
            input_size,
            hidden_size,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            device=device,
            num_layers=num_layers,
        )
        self.mdn = nn.Linear(hidden_size, (2 * input_size + 1) * gaussians + 2)

    def forward(self, latent_states):
        """
        latent_states: (B, S, Z_dim+compl)
        actions: (B, S, a_size)
        """
        # inp = torch.cat([latent_states, actions], axis=-1)
        inp = latent_states
        outp = self.lstm(inp)[:, -1, :]  # (B, LSTM_hidden_size)
        gmm_out = self.mdn(outp)
        stride = self.gaussians * latent_states.shape[2]
        mus = gmm_out[:, :stride]
        mus = mus.view(
            latent_states.shape[0],
            # latent_states.shape[1],
            self.gaussians,
            latent_states.shape[-1],
        )

        # mus -> [B, n_gaussians, Z_dim+Compl]

        sigmas = gmm_out[:, stride : 2 * stride]
        sigmas = sigmas.view(
            latent_states.shape[0],
            # latent_states.shape[1],
            self.gaussians,
            latent_states.shape[-1],
        )
        sigmas = torch.exp(sigmas)

        pi = gmm_out[:, 2 * stride : 2 * stride + self.gaussians]
        pi = pi.view(latent_states.shape[0], self.gaussians)
        log_pi = f.log_softmax(pi, dim=-1)

        if self.mode == "training":
            rs = gmm_out[:, -2]
            ds = gmm_out[:, -1]
            return mus, sigmas, log_pi, rs, ds
        elif self.mode == "inference":
            w = Categorical(logits=log_pi).sample()  # (B, S) [1]
            next_latent_pred = Normal(mus, sigmas).sample()[range(w.shape[0]), w, :]
            # (B, Z_dim+Complt)
            return next_latent_pred

    def gmm_loss(self, next_latent, mus, sigmas, log_pi, a="no_max"):

        """
        next_latent_obs : (B, S, Z_dim+Compl)
        mus: (B, S, n_gaussians, Z_dim+Compl)
        sigmas: (B, S, n_gaussians, Z_dim+Compl)
        log_pi: (B, S, n_gaussians)
        """
        next_latent = next_latent.unsqueeze(-2)  # [B, S, 1, Z_dim+Complt]
        normal_dist = Normal(mus, sigmas)
        log_probs = normal_dist.log_prob(next_latent)  # [B, S, n_g, Z_dim+Complt]
        log_probs = log_pi + log_probs.sum(dim=-1)  # (B, S, n_g)

        if a == "no_max":
            log_prob = log_probs.sum(dim=-1)
        else:
            max_log_probs = log_probs.max(dim=-1, keepdim=True)[0]  # (B, S, 1)
            log_probs = log_probs - max_log_probs  # ??
            probs = log_probs.exp()
            probs = probs.sum(dim=-1)  # (B, S)
            log_prob = max_log_probs.squeeze() + probs
        return -log_prob  # (B, S)

    def get_loss(self, gmm_loss, rewards=None, rs=None, terminals=None, ds=None):

        bce = 0
        scale = self.input_size
        if terminals is not None and ds is not None:
            ds = ds.exp() > 0.5
            bce = f.binary_cross_entropy(ds.float(), terminals.float())
            scale += 1
        mse = 0
        if rewards is not None and rs is not None:
            mse = f.mse_loss(rewards, rs)
            scale += 1
        loss = (gmm_loss + bce + mse) / scale
        return loss


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        seq_len,
        batch_size,
        device,
        num_layers=1,
        training_initial_hidden_states=False,
    ):

        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.device = device
        self.training_initial_hidden_states = training_initial_hidden_states
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        if training_initial_hidden_states:
            # assume unidirectional
            self.c_0 = torch.empty(
                (self.num_layers * 1, self.batch_size, self.hidden_size),
                requires_grad=True,
            )
            self.h_0 = torch.empty(
                (self.num_layers * 1, self.batch_size, self.hidden_size),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.h_0)
            nn.init.xavier_normal_(self.c_0)

            self.c_0 = self.c_0.to(self.device)
            self.h_0 = self.h_0.to(self.device)

            self.h_n, self.c_n = None, None

    def forward(self, input):
        """
        :param input: (Tensor: [B, S, input_size])
        :param hidden_state: (Tensor: [B, S, Hidden_size])
        :return h_n: (Tensor: [B, Bidirectional*Num_layers, Hidden_size])
        """
        if self.training_initial_hidden_states:
            outp, (_, _) = self.lstm(input, (self.h_0, self.c_0))
        else:
            outp, (_, _) = self.lstm(input)  # initialize to zeros
        return outp

    def reset_lstm(self):
        self.h_n, self.c_n = None, None


if __name__ == "__main__":

    lstm = LSTM(
        input_size=8,
        hidden_size=12,
        seq_len=2,
        batch_size=4,
        device="cpu",
        num_layers=1,
        training_initial_hidden_states=False,
    )

    opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)
    for i in range(2):
        input = torch.rand((4, 2, 8))
        outp = lstm(input)
        print(outp.shape)

        # loss = torch.tensor([1.2])
        loss = outp.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
