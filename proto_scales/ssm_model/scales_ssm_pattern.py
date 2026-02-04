import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import proto_scales.ssm_model.scales_ssm as scales_ssm



class DeepSSMPatternConditioned(nn.Module):
    """
    q(z_t | y_{1:t}, u_{1:t}) via GRU on [y,u]
    p(z_t | z_{t-1}, u_t) via MLP([z_{t-1},u_t]) -> (mu, logvar)
    p(y_t | z_t) via MLP(z_t) -> (mu_y) with learned global sigma_y
    """
    def __init__(self, y_dim, u_dim, z_dim=16, rnn_hidden=62, mlp_hidden=128, emission_uses_u=False):
        super().__init__()
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.emission_uses_u = emission_uses_u

        self.gru = nn.GRU(input_size=y_dim + u_dim, hidden_size=rnn_hidden, batch_first=True)
        self.q_head = nn.Linear(rnn_hidden, 2 * z_dim)

        self.trans = scales_ssm.MLP(z_dim + u_dim, 2 * z_dim, hidden=mlp_hidden)

        emit_in = z_dim + (u_dim if emission_uses_u else 0)
        self.emit = scales_ssm.MLP(emit_in, y_dim, hidden=mlp_hidden)

        #pattern scaling like head to for emission
        self.ctrl_lin = nn.Linear(u_dim, y_dim,bias=True)

        # global observation noise (log sigma); initialized modestly
        self.log_sigma_y = nn.Parameter(torch.tensor(-0.2))

    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def forward_elbo(self, y, u, kl_free_bits=0.5):
        """
        y,u: [B, T, Dy/Du]
        Returns loss, stats.
        """
        B, T, _ = y.shape

        rnn_in = torch.cat([y, u], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h)
        mu_q, logvar_q = torch.chunk(q_params, 2, dim=-1)

        # stabilize logvar range...numbers not tuned
        logvar_q = torch.clamp(logvar_q, -12.0, 6.0)

        # prior for z0
        mu_p0 = torch.zeros(B, self.z_dim, device=y.device)
        logvar_p0 = torch.zeros(B, self.z_dim, device=y.device)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        nll = 0.0
        kl = 0.0

        z_prev = None
        for t in range(T):
            z_t = self.sample(mu_q[:, t], logvar_q[:, t])
            ctrl = self.ctrl_lin(u[:, t])              # [B, y_dim]
                    
            if self.emission_uses_u:
                res = self.emit(torch.cat([z_t, u[:, t]], dim=-1))  # [B, y_dim]  (residual from latent)
            else:
                res = self.emit(z_t)  # [B, y_dim]  (residual from latent)
            
            y_hat = ctrl + res

            # Gaussian NLL (includes log sigma term)
            nll_t = 0.5 * (((y[:, t] - y_hat) / sigma_y) ** 2).sum(-1) + self.y_dim * torch.log(sigma_y)
            nll = nll + nll_t

            if t == 0:
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, 0], logvar_q[:, 0], mu_p0, logvar_p0)
            else:
                trans_in = torch.cat([z_prev, u[:, t]], dim=-1)
                mu_p, logvar_p = torch.chunk(self.trans(trans_in), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                kl_t = scales_ssm.diag_gaussian_kl(mu_q[:, t], logvar_q[:, t], mu_p, logvar_p)

            # free-bits: don't over-penalize small KL; helps avoid posterior collapse: from Claude
            # Applied per-sample
            kl = kl + torch.clamp(kl_t, min=kl_free_bits)

            z_prev = z_t

        # per-batch means
        nll = nll.mean()
        kl = kl.mean()
        return nll, kl

    @torch.no_grad()
    def forecast(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        ysamps = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                ctrl = self.ctrl_lin(u_t)

                if self.emission_uses_u:
                    res = self.emit(torch.cat([z, u_t], dim=-1))
                else:
                    res = self.emit(z)
                
                y_hat = ctrl + res

                y_s = y_hat + sigma_y * torch.randn_like(y_hat)
                preds.append(y_s)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0)

    @torch.no_grad()
    def forecast_deterministic(self, y_ctx, u_ctx, u_fut, steps, n_samples=50):
        B, Tc, _ = y_ctx.shape

        rnn_in = torch.cat([y_ctx, u_ctx], dim=-1)
        h, _ = self.gru(rnn_in)
        q_params = self.q_head(h[:, -1:])
        mu_qT, logvar_qT = torch.chunk(q_params.squeeze(1), 2, dim=-1)
        logvar_qT = torch.clamp(logvar_qT, -12.0, 6.0)

        sigma_y = torch.exp(torch.clamp(self.log_sigma_y, -6.0, 3.0))

        ysamps = []
        for _ in range(n_samples):
            z = self.sample(mu_qT, logvar_qT)

            preds = []
            for k in range(steps):
                u_t = u_fut[:, k]
                mu_p, logvar_p = torch.chunk(self.trans(torch.cat([z, u_t], dim=-1)), 2, dim=-1)
                logvar_p = torch.clamp(logvar_p, -12.0, 6.0)
                z = self.sample(mu_p, logvar_p)

                if self.emission_uses_u:
                    y_hat = self.emit(torch.cat([z, u_t], dim=-1))
                else:
                    y_hat = self.emit(z)

                y_s = y_hat #+ sigma_y * torch.randn_like(y_hat)
                preds.append(y_s)

            ysamps.append(torch.stack(preds, dim=1))  # [B,steps,Dy]

        samp = torch.stack(ysamps, dim=0)  # [S,B,steps,Dy]
        return samp.mean(0), samp.quantile(0.10, 0), samp.quantile(0.90, 0)


# -------------------------
# Train / eval loop with early stopping
# -------------------------
def run_train(
    y_np, u_np,
    context_len=40, horizon=12,
    batch_size=64,
    epochs=50,
    lr=2e-3,
    z_dim=16,
    device="cpu",
):
    # split
    N = y_np.shape[0]
    idx = np.random.permutation(N)
    n_train = int(0.8 * N)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    y_tr, u_tr = y_np[tr_idx], u_np[tr_idx]
    y_va, u_va = y_np[va_idx], u_np[va_idx]

    # normalize (fit on train only)
    y_scaler = scales_ssm.StandardScaler().fit(y_tr)
    u_scaler = scales_ssm.StandardScaler().fit(u_tr)
    y_trn = y_scaler.transform(y_tr)
    y_van = y_scaler.transform(y_va)
    u_trn = u_scaler.transform(u_tr)
    u_van = u_scaler.transform(u_va)
   
    train_ds = scales_ssm.UnifiedWindowDataset(y_trn, u_trn, context_len=context_len, horizon=horizon)
    val_ds   = scales_ssm.UnifiedWindowDataset(y_van, u_van, context_len=context_len, horizon=horizon)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    Dy = y_np.shape[-1]
    Du = u_np.shape[-1]
    model = DeepSSMPatternConditioned(y_dim=Dy, u_dim=Du, z_dim=z_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = float("inf")
    best_state = None
    patience, patience_left = 15, 15
  
    # KL annealing schedule: ramp from 0 -> 1 over first ~30% of training
    total_steps = epochs * len(train_dl)
    global_step = 0

    for epoch in range(1, epochs + 1):
       
        model.train()
        
        tr_loss = []

        for y_ctx, u_ctx, u_fut, y_fut in train_dl:
           
            y_ctx = torch.tensor(y_ctx, device=device)
            u_ctx = torch.tensor(u_ctx, device=device)
            u_fut = torch.tensor(u_fut, device=device)
            y_fut = torch.tensor(y_fut, device=device)
          
            # We train on full (context+horizon) to teach dynamics across the boundary:
            y_full = torch.cat([y_ctx, y_fut], dim=1)
            u_full = torch.cat([u_ctx, u_fut], dim=1)
       
            nll, kl = model.forward_elbo(y_full, u_full, kl_free_bits=0.2)
            mean, _, _ = model.forecast_deterministic(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
            roll_out_mse =((mean - y_fut) ** 2).mean()
           
            # anneal KL weight
            global_step += 1
            frac = min(1.0, global_step / int(0.3 * total_steps))
            kl_w = frac  # 0->1
            kl_w = 5
            alpha = 10000
         
            loss = nll + kl_w * kl + alpha*roll_out_mse

            if(global_step%100==0):
                print("loss: ",nll.item(),kl_w,kl.item(),roll_out_mse.item())
            
            opt.zero_grad()
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
           
            opt.step()
           

            tr_loss.append(loss.item())

        # validation: one-step objective + forecast MSE on horizon
        model.eval()
        
        va_loss = []
        va_mse = []

        with torch.no_grad():
            for y_ctx, u_ctx, u_fut, y_fut in val_dl:
                
                y_ctx = torch.tensor(y_ctx, device=device)
                u_ctx = torch.tensor(u_ctx, device=device)
                u_fut = torch.tensor(u_fut, device=device)
                y_fut = torch.tensor(y_fut, device=device)

                y_full = torch.cat([y_ctx, y_fut], dim=1)
                u_full = torch.cat([u_ctx, u_fut], dim=1)

                nll, kl = model.forward_elbo(y_full, u_full, kl_free_bits=0.2)
                loss = nll + 1.0 * kl
                va_loss.append(loss.item())

                mean, _, _ = model.forecast(y_ctx, u_ctx, u_fut, steps=horizon, n_samples=30)
                mse = ((mean - y_fut) ** 2).mean().item()
                va_mse.append(mse)

        tr = float(np.mean(tr_loss))
        va = float(np.mean(va_loss))
        mse = float(np.mean(va_mse))
        print(f"epoch {epoch:03d} | train {tr:.4f} | val_elbo {va:.4f} | val_mse {mse:.4f}")
        os.makedirs("outputs_ssm_scales", exist_ok=True)
        torch.save(model.state_dict(),"outputs_ssm_scales/model_out")

        # early stopping on val_elbo
        if va < best_val - 1e-4:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, y_scaler, u_scaler
