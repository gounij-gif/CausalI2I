import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime

np.random.seed(42)
if np.random.choice(np.arange(1000)) != 102:
    raise ValueError("Random seed is not set correctly.")

def batch_iterator(u, i, y, batch_size):
    N = u.size(0)
    for start in range(0, N, batch_size):
        end = start + batch_size
        yield u[start:end], i[start:end], y[start:end]

def success_metrics(logits, ground_truth, threshold=0.0):

    # work in float32 on the same device
    scores = logits.detach().float()
    gt = (ground_truth.detach() > 0.5)

    # binary predictions from logits threshold
    preds = scores > float(threshold)

    # counts
    true = gt.sum()
    positive = preds.sum()
    true_positive = (preds & gt).sum()

    # precision/recall (safe divide)
    precision = (true_positive.float() / positive.clamp(min=1).float()).item()
    recall    = (true_positive.float() / true.clamp(min=1).float()).item()

    # MPR via double argsort
    ranks = torch.argsort(torch.argsort(scores))
    denom = max(scores.numel() - 1, 1)
    pr = ranks.float() / denom
    mpr = pr[gt].mean().item() if gt.any() else 0.0

    return {'precision': precision, 'recall': recall, 'mpr': mpr}

class MatrixFactorizationTorch(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_factors  = n_factors
        self.lr         = None
        self.wd         = None
        self.pos_weight = None
        self.batch_size = None
        self.n_epochs   = None
        self.device     = None
        self.use_amp    = None
        self.note       = None
        
        self.metrics    = {
            'Epoch': [], 
            'Train': {'BCE': [], 'BCE-POS': [], 'BCE-NEG': [], 'Precision': [], 'Recall': [], 'MPR': []},
            'Validation': {'BCE': [], 'BCE-POS': [], 'BCE-NEG': [], 'Precision': [], 'Recall': [], 'MPR': []},
            'Epoch\'s Change': [],
            'COS θ' : [],
            'Elapsed': []
        }
        
        # biases + embeddings
        self.mu = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))  # global bias
        self.b_u = torch.nn.Parameter(torch.zeros(n_users + 1, dtype=torch.float32))
        self.b_i = torch.nn.Parameter(torch.zeros(n_items + 1, dtype=torch.float32))
        self.P   = torch.nn.Parameter(torch.randn(n_users + 1,  n_factors, dtype=torch.float32) * 0.1)
        self.Q   = torch.nn.Parameter(torch.randn(n_items + 1,  n_factors, dtype=torch.float32) * 0.1)

    def predict_logit(self, u_idx, i_idx):
        # returns raw score = logit
        return (
            self.mu
            + self.b_u[u_idx]
            + self.b_i[i_idx]
            + (self.P[u_idx] * self.Q[i_idx]).sum(dim=1)
        )
    
    def predict_prob(self, u_idx, i_idx):
        # returns probability
        logits = self.predict_logit(u_idx, i_idx)
        return torch.sigmoid(logits)
    
    def batched_logits(self, u, i):
        out = []
        for start in range(0, u.size(0), self.batch_size):
            end = start + self.batch_size
            out.append(self.predict_logit(u[start:end], i[start:end]))
        return torch.cat(out, dim=0)

    def fit(self, train_data, val_data, lr, wd, pos_weight, batch_size, n_epochs, device=None, use_amp=False):
        print("Initializing training...", end='\r')
        
        # input parameters
        self.lr         = lr
        self.wd         = wd
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.n_epochs   = n_epochs
        self.device     = device
        self.use_amp    = use_amp
        n_batches = len(train_data) // batch_size + 1

        # device, optimizer, scheduler, loss function
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        is_cuda = (device.type == 'cuda')
        optimizer = torch.optim.Adam([
            {'params': [self.mu], 'weight_decay': 0.0},
            {'params': [self.b_i, self.b_u], 'weight_decay': wd * 0.1},
            {'params': [self.Q, self.P], 'weight_decay': wd}
        ], lr=lr)
        bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, dtype=torch.float32),
            reduction='none'
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and is_cuda)

        # --- move data to tensors ---
        # training data full tensor (for eval)
        u_tr = torch.tensor(train_data[:,0].astype(int), dtype=torch.long, device=device)
        i_tr = torch.tensor(train_data[:,1].astype(int), dtype=torch.long, device=device)
        y_tr = torch.tensor(train_data[:,2].astype(int), dtype=torch.float32, device=device)
        pos_mask_tr = (y_tr == 1)
        neg_mask_tr = (y_tr == 0)


        # val data full tensor
        u_val = torch.tensor(val_data[:,0].astype(int), dtype=torch.long, device=device)
        i_val = torch.tensor(val_data[:,1].astype(int), dtype=torch.long, device=device)
        y_val = torch.tensor(val_data[:,2].astype(int), dtype=torch.float32, device=device)
        pos_mask_val = (y_val == 1)
        neg_mask_val = (y_val == 0)

        # --- print title ----
        print("Epoch  ||- - - - - - - - Train - - - - - - - -||- - - - - - Validation - - - - - - - || Epoch's | COS θ | Time     ")
        print("Number || BCE    | BCE-POS | BCE-NEG | MPR    || BCE    | BCE-POS | BCE-NEG | MPR    || Change  |       | Elapsed  ")
        print("=======||========|=========|=========|========||========|=========|=========|========||=========|=======|==========")

        # --- training loop ---
        start_time = datetime.now()
        for epoch in range(1, n_epochs+1):

            # record old parameters for norm calculation
            old_params = torch.cat([p.data.flatten() for p in self.parameters()])

            # shuffle the training data
            g = torch.Generator(device=device)
            g.manual_seed(epoch)
            perm = torch.randperm(u_tr.size(0), generator=g, device=device)
            u_epoch = u_tr[perm]
            i_epoch = i_tr[perm]
            y_epoch = y_tr[perm]

            # batch training
            batch = 1
            for u_b, i_b, y_b in batch_iterator(u_epoch, i_epoch, y_epoch, batch_size=batch_size):
                print(f" {epoch:>3}  .... Training: Batch {batch} out of {n_batches} ({batch/n_batches:.2%})       ", end='\r')
                batch += 1

                optimizer.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast('cuda' if is_cuda else 'cpu', enabled=(use_amp and is_cuda))                
                with amp_ctx:
                    logits = self.predict_logit(u_b, i_b)
                    loss = bce(logits, y_b).mean()

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # calculate norms of parameters
            new_params = torch.cat([p.data.flatten() for p in self.parameters()])
            step_vec = new_params - old_params
            upd_norm = torch.norm(step_vec).item()

            # cosine similarity between prev_step and current step_vec
            if 'prev_step' in locals():
                cos_sim = (prev_step @ step_vec) / (prev_step.norm() * step_vec.norm() + 1e-12)
                cos_sim = f"{cos_sim:.3f}" if cos_sim > 0 else f"{cos_sim:.2f}"
            else:
                cos_sim = "None "
            prev_step = step_vec.clone()

            # end of epoch: full evaluation on train & val
            amp_ctx = torch.amp.autocast('cuda' if is_cuda else 'cpu', enabled=(use_amp and is_cuda))
            with torch.no_grad(), amp_ctx:
                # train
                logits_tr = self.batched_logits(u_tr, i_tr)
                train_losses = bce(logits_tr, y_tr)
                train_loss = train_losses.mean().item()
                train_losses_pos = train_losses[pos_mask_tr].mean().item()
                train_losses_neg = train_losses[neg_mask_tr].mean().item()
                success_tr = success_metrics(logits_tr, y_tr)
                # val
                logits_v = self.batched_logits(u_val, i_val)
                val_losses = bce(logits_v, y_val)
                val_loss = val_losses.mean().item()
                val_losses_pos = val_losses[pos_mask_val].mean().item()
                val_losses_neg = val_losses[neg_mask_val].mean().item()
                success_v = success_metrics(logits_v, y_val)
                
            elapsed = (datetime.now() - start_time).total_seconds()
            
            # print results
            if True:
                print(f" {epoch:>3}   ||",
                        f"{train_loss:.4f} | ",
                        f"{train_losses_pos:.4f} | ",
                        f"{train_losses_neg:.4f} |",
                        f"{success_tr['mpr']:.4f} ||",
                        f"{val_loss:.4f} | ",
                        f"{val_losses_pos:.4f} | ",
                        f"{val_losses_neg:.4f} |",
                        f"{success_v['mpr']:.4f} ||",
                        f"{upd_norm:>6.2f}  |",
                        cos_sim, "|",
                        f"{int(elapsed//60):02}:{elapsed%60:05.2f}"
                )

            # store success metrics
            if True:
                # store metrics for plotting
                self.metrics['Epoch'].append(epoch)
                for split, success in [('Train', success_tr), ('Validation', success_v)]:
                    self.metrics[split]['BCE'].append(train_loss if split == 'Train' else val_loss)
                    self.metrics[split]['BCE-POS'].append(train_losses_pos if split == 'Train' else val_losses_pos)
                    self.metrics[split]['BCE-NEG'].append(train_losses_neg if split == 'Train' else val_losses_neg)
                    self.metrics[split]['Precision'].append(success['precision'])
                    self.metrics[split]['Recall'].append(success['recall'])
                    self.metrics[split]['MPR'].append(success['mpr'])
                self.metrics['Epoch\'s Change'].append(upd_norm)
                self.metrics['Elapsed'].append(elapsed)

    def save(self, path: str, note: str = None):
        """
        Save model state, metrics, and an optional description to a file.
        """
        self.note = note
        torch.save({
            'state_dict': self.state_dict(),
            'lr': self.lr,
            'wd': self.wd,
            'pos_weight': self.pos_weight,
            'batch_size': self.batch_size,
            'n_epochs': self.n_epochs,
            'device': self.device,
            'use_amp': self.use_amp,
            'metrics': self.metrics,
            'timestamp': f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'note': self.note,
        }, path)

    def load(self, path: str):
        """
        Load model state, metrics, and description from a file into this instance.
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.lr = checkpoint.get('lr', None)
        self.wd = checkpoint.get('wd', None)
        self.pos_weight = checkpoint.get('pos_weight', None)
        self.batch_size = checkpoint.get('batch_size', None)
        self.n_epochs = checkpoint.get('n_epochs', None)
        self.device = checkpoint.get('device', None)
        self.use_amp = checkpoint.get('use_amp', None)
        self.metrics = checkpoint.get('metrics', {})
        self.note = checkpoint.get('note', None)

        # Print loaded model summary
        print("Loaded model summary:")
        print(f"Model:                      MatrixFactorizationTorch")
        print(f"Number of users:            {self.n_users}")
        print(f"Number of items:            {self.n_items}")
        print(f"Number of factors:          {self.n_factors}")
        print(f"Learning rate:              {self.lr}")
        print(f"Weight decay:               {self.wd}")
        print(f"Positive weight:            {self.pos_weight}")
        print(f"Batch size:                 {self.batch_size}")
        print(f"Number of epochs:           {self.n_epochs}")
        print(f"Device:                     {self.device}")
        print(f"Use AMP:                    {self.use_amp}")
        print(f"Timestamp:                  {checkpoint.get('timestamp', 'N/A')}")
        if self.note:
            print(f"Note:                       {self.note}")

    def plot(
            self, 
            prediction_metrics=True, 
            bce_pos=True,
            pred_vs_act=True,
            correlations=True,
            biases=True,
            norms=True,
            coordinates=True,
            first_epoch=1):
        
        device = next(self.parameters()).device

        def t2list(t):
            return t.detach().to('cpu').flatten().tolist()
        
        epochs = self.metrics['Epoch']
        precisions = np.array(self.metrics['Validation']['Precision'])
        recalls = np.array(self.metrics['Validation']['Recall'])

        mprs = np.array(self.metrics['Validation']['MPR'])

        if first_epoch > 1:
            epochs = epochs[first_epoch-1:]
            precisions = precisions[first_epoch-1:]
            recalls = recalls[first_epoch-1:]
            mprs = mprs[first_epoch-1:]

        if prediction_metrics:
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, precisions, marker='o', markersize=3, label='Precision')
            plt.plot(epochs, recalls, marker='o', markersize=3, label='Recall')
            plt.plot(epochs, mprs, marker='o', markersize=3, label='MPR')

            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.title('Validation Metrics Over Epochs')
            plt.legend()
            plt.grid()
            plt.show()

        if bce_pos:
            bce_pos_train = np.array(self.metrics['Train']['BCE-POS'])
            bce_pos_val = np.array(self.metrics['Validation']['BCE-POS'])
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, bce_pos_train, marker='o', markersize=3, label='Train BCE-POS')
            plt.plot(epochs, bce_pos_val, marker='o', markersize=3, label='Validation BCE-POS')
            plt.ylim(0)
            plt.xlabel('Epoch')
            plt.ylabel('BCE-POS Loss')
            plt.title('BCE-POS Loss Over Epochs')
            plt.legend()
            plt.grid()
            plt.show()
            
        if pred_vs_act:
            baseline = test['rating'].mean()
            bottom = (1 - recalls) * baseline
            positive = (recalls * baseline / np.maximum(precisions, 1e-4))
            top = positive + bottom
            plt.figure(figsize=(8, 4))
            plt.fill_between(epochs, bottom, top, color='lightblue', alpha=0.7, label='Predicted Positive')
            plt.fill_between(epochs, 0, baseline, color='red', alpha=0.3, label='Actually Positive')
            plt.ylim(0)
            plt.xlim(first_epoch, max(epochs))
            plt.title('Predicted vs Actual Positive Ratings')
            plt.xlabel('Epoch')
            plt.ylabel('Rating')
            plt.legend()
            plt.grid()
            plt.show()

        if correlations:
            test_copy = test.copy()
            with torch.no_grad():
                u_idx = torch.tensor(test_copy['user_idx'].values, dtype=torch.long, device=device)
                i_idx = torch.tensor(test_copy['item_idx'].values, dtype=torch.long, device=device)
                amp_ctx = torch.amp.autocast('cuda', enabled=(device.type == 'cuda'), dtype=torch.float16) if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    probs = self.predict_prob(u_idx, i_idx)
            test_copy['probability'] = t2list(probs) 
            users_good = test_copy.groupby('user_idx').filter(
                lambda g: g['rating'].nunique()>1 and g['probability'].nunique()>1
            )
            users_corr = users_good.groupby('user_idx').apply(lambda x: x['rating'].corr(x['probability']), include_groups=False)
            items_good = test_copy.groupby('item_idx').filter(
                lambda g: g['rating'].nunique()>1 and g['probability'].nunique()>1
            )
            items_corr = items_good.groupby('item_idx').apply(lambda x: x['rating'].corr(x['probability']), include_groups=False)

            plt.figure(figsize=(8, 4))
            plt.hist(users_corr.dropna(), bins=50, alpha=0.5, label='User Correlation')
            plt.hist(items_corr.dropna(), bins=50, alpha=0.5, label='Item Correlation')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Frequency')
            plt.title('Correlation of Ratings and Predicted Probabilities')
            plt.legend()
            plt.show()

        if biases:
            b_u = t2list(self.b_u)
            b_i = t2list(self.b_i)
            plt.figure(figsize=(8, 4))
            plt.hist(b_u, bins=80, alpha=0.5, label='User Bias (b_u)')
            plt.hist(b_i, bins=80, alpha=0.5, label='Item Bias (b_i)')
            plt.legend()
            plt.xlabel('Bias values')
            plt.ylabel('Frequency')
            plt.title('Distribution of Bias Values')
            plt.show()

        if norms:
            norms_P = torch.norm(self.P.detach(), dim=1)
            norms_Q = torch.norm(self.Q.detach(), dim=1)
            norms_P = t2list(norms_P)
            norms_Q = t2list(norms_Q)

            plt.figure(figsize=(8, 4))
            plt.hist(norms_P, bins=100, alpha=0.5, label='Norms of P (Users)')
            plt.hist(norms_Q, bins=100, alpha=0.5, label='Norms of Q (Items)')
            plt.xlabel('Norm values')
            plt.ylabel('Frequency')
            plt.title('Distribution of Norms of P and Q')
            plt.legend()
            plt.show()

        if coordinates:
            P_lower = torch.quantile(self.P, 0.05, dim=0)
            P_upper = torch.quantile(self.P, 0.95, dim=0)
            P_mean  = self.P.mean(dim=0)
            Q_lower = torch.quantile(self.Q, 0.05, dim=0)
            Q_upper = torch.quantile(self.Q, 0.95, dim=0)
            Q_mean  = self.Q.mean(dim=0)

            P_base   = t2list(P_lower)
            P_height = t2list(P_upper - P_lower)
            P_mean   = t2list(P_mean)
            Q_base   = t2list(Q_lower)
            Q_height = t2list(Q_upper - Q_lower)
            Q_mean   = t2list(Q_mean)

            x = np.arange(1, self.n_factors + 1)
            w = 0.35
            plt.figure(figsize=(10, 5))
            plt.bar(x + w/2, P_height, bottom=P_base, width=w, alpha=0.8, label='Users')
            plt.bar(x - w/2, Q_height, bottom=Q_base, width=w, alpha=0.8, label='Items')
            plt.scatter(x+w/2, P_mean, color='green', marker='o')
            plt.scatter(x-w/2, Q_mean, color='red', marker='o')
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xticks(x)
            plt.xlabel("Column index")
            plt.ylabel("Value")
            plt.title("90% Intervals for P and Q")
            plt.legend()
            plt.grid(axis='y')
            plt.show()