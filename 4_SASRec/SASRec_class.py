import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime

import random, numpy as np

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SASRecTorch(nn.Module):
    """
    PyTorch implementation of SASRec for implicit-feedback sequential recommendation.
    Includes an integrated .fit() method for training with next-K window BCE loss.
    """

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        d_model: int = 128,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        device: str = "cpu",
    ):
        super().__init__()

        fix_seed(42)
        
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.padding_idx = num_items

        # === Embeddings === #
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=self.padding_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # === Transformer encoder === #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection (good for generalization)
        self.output_layer = nn.Linear(d_model, d_model, bias=False)

        nn.init.normal_(self.item_embedding.weight, std=d_model ** -0.5)

        self.to(device)

    # -------------------------------------------------------------------------
    # --- MODEL COMPONENTS ---
    # -------------------------------------------------------------------------

    def forward(self, seq):
        """
        seq: (batch, L) of padded item indices.
        Returns:
            h: (batch, L, d_model) hidden states for each position.
        """
        batch, L = seq.size()
        device = seq.device
        pos = torch.arange(L, device=device).unsqueeze(0).expand(batch, L)
        x = self.item_embedding(seq) + self.pos_embedding(pos)

        # causal mask: position t cannot see positions > t
        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        h = self.encoder(x, mask=mask)             # (batch, L, d_model)
        return h

    def predict_scores(self, seq, candidates):
        """
        seq: (batch, L)
        candidates: (batch, C) or (C,)
        Returns:
            scores: (batch, C)
        """
        if candidates.dim() == 1:
            candidates = candidates.unsqueeze(0).expand(seq.size(0), -1)

        h = self.forward(seq)                     # (batch, L, d_model)
        last_state = h[:, -1, :]                  # (batch, d_model)

        cand_emb = self.item_embedding(candidates)  # (batch, C, d_model)

        scores = (last_state.unsqueeze(1) * cand_emb).sum(-1)  # (batch, C)
        return scores
    
    def sasrec_nextitem_loss(self, seq):
        """
        SASRec loss:
        For each timestep t, predict next item with
        BCE on (positive, 1 negative).
        """
        device = seq.device
        # batch, L = seq.size()

        # ---- Forward pass ----
        h = self.forward(seq)                        # (B, L, d_model)

        # inputs are positions 0..L-2
        h_in = h[:, :-1, :]                           # (B, L-1, d)
        pos_items = seq[:, 1:]                        # (B, L-1)

        # mask padding
        valid = (seq[:, :-1] != self.padding_idx) & (pos_items != self.padding_idx)

        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # flatten valid states
        h_flat = h_in[valid]                           # (N, d)
        pos_flat = pos_items[valid]                    # (N,)

        # ---- Positive logits ----
        pos_emb = self.item_embedding(pos_flat)        # (N, d)
        pos_logits = (h_flat * pos_emb).sum(-1)        # (N,)

        # ---- Sample 1 negative per positive ----
        # uniform sampling; efficient on GPU
        # you can optionally re-sample when neg == pos, but statistically negligible
        num_items = self.num_items
        neg_samples = torch.randint(
            low=0, high=num_items,
            size=pos_flat.shape,
            device=device
        )

        neg_emb = self.item_embedding(neg_samples)
        neg_logits = (h_flat * neg_emb).sum(-1)

        # ---- BCE loss ----
        # L = -log(sigmoid(pos)) - log(1 - sigmoid(neg))
        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction='mean'
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction='mean'
        )

        return loss_pos + loss_neg

    # -------------------------------------------------------------------------
    # --- FIT METHOD WITH MF-STYLE LOGGING ---
    # -------------------------------------------------------------------------

    def fit(
        self,
        train_dataset,        # expected: numpy array or torch tensor, shape (N, max_seq_len)
        valid_dataset=None,   # same
        batch_size=128,
        lr=1e-3,
        weight_decay=0.0,
        num_epochs=20,
    ):
        """
        Train SASRec using (N, L) arrays as input.
        Each row is a padded sequence of item IDs (0 is padding).
        """

        fix_seed(42)

        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        # ------- Convert raw arrays to PyTorch datasets -------- #
        # train_dataset: (N, L)
        train_tensor = torch.as_tensor(train_dataset, dtype=torch.long)
        train_loader = DataLoader(
            train_tensor, 
            batch_size=batch_size, 
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
            generator=torch.Generator().manual_seed(42))

        if valid_dataset is not None:
            valid_tensor = torch.as_tensor(valid_dataset, dtype=torch.long)
            valid_loader = DataLoader(valid_tensor, batch_size=batch_size)
        else:
            valid_loader = None

        # ------- Optimizer -------- #
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        n_batches = len(train_loader)

        print("Epoch | T-Loss | V-Loss | Pctl  | HR10  | NDCG  | Cosθ  | Elapsed Time")
        print("======|========|========|=======|=======|=======|=======|=============")

        from datetime import datetime
        start_time = datetime.now()

        prev_step_vec = None

        # ====================================================== #
        # ===================== EPOCH LOOP ===================== #
        # ====================================================== #
        for epoch in range(1, num_epochs + 1):

            old_params = torch.cat([p.data.flatten() for p in self.parameters()])

            self.train()
            total_loss = 0.0
            batch_id = 1

            # ---------------- TRAINING ---------------- #
            for seq_batch in train_loader:
                print(
                    f"... Training: Batch {batch_id}/{n_batches} ({batch_id/n_batches:.1%})        ",
                    end="\r",
                )
                batch_id += 1

                seq_batch = seq_batch.to(self.device)

                optimizer.zero_grad()

                loss = self.sasrec_nextitem_loss(seq_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches

            # ---------------- UPDATE DIRECTION COSINE ---------------- #
            new_params = torch.cat([p.data.flatten() for p in self.parameters()])
            step_vec = new_params - old_params

            if prev_step_vec is not None:
                cosθ = (
                    (prev_step_vec @ step_vec)
                    / (prev_step_vec.norm() * step_vec.norm() + 1e-12)
                )
                cosθ = f"{cosθ:.3f}"
            else:
                cosθ = "None "
            prev_step_vec = step_vec.clone()

            # ---------------- VALIDATION ---------------- #
            with torch.no_grad():
                self.eval()
                val_total = 0.0
                metrics_accum = {
                    "percentile": [],
                    "HR10": [],
                    "NDCG": [],
                }

                val_batches = len(valid_loader)
                val_id = 1

                for seq_batch in valid_loader:
                    print(
                        f"... Validating: Batch {val_id}/{val_batches} ({val_id/val_batches:.1%})        ",
                        end="\r",
                    )
                    val_id += 1
                    seq_batch = seq_batch.to(self.device)

                    # BCE loss
                    loss = self.sasrec_nextitem_loss(seq_batch)
                    val_total += loss.item()

                    # ---------- Classification-based metrics (Paper: 100 negatives) ----------
                    seq_eval = seq_batch[:, :-1]                                     # (B,L-1)
                    true_eval = seq_batch[:, -1]                                     # (B,)

                    # candidates: true item + 100 random negatives
                    B = seq_batch.size(0)
                    random_sample = torch.randint(0, self.num_items, (B, 100), device=self.device)
                    candidates = torch.cat([true_eval.unsqueeze(1), random_sample], dim=1)     # (B,101)

                    # ====== Compute logits efficiently ======
                    h = self.forward(seq_eval)                                       # (B,L,d)
                    h_last = h[:, -1, :]                                             # (B,d)
                    cand_emb = self.item_embedding(candidates)                       # (B,101,d)
                    logits = (h_last.unsqueeze(1) * cand_emb).sum(-1)                # (B,101)

                    # ====== Ranking ======
                    sorted_idx = torch.argsort(logits, dim=1, descending=True)
    
                    # HR@10
                    hit = (sorted_idx[:, :10] == 0).any(dim=1).float().mean().item()

                    # NDCG
                    true_ranks = (sorted_idx == 0).nonzero()[:, 1].float()           # (B,)
                    ndcg = (1.0 / torch.log2(true_ranks + 2)).mean().item()

                    # Percentile
                    percentile = (1 - true_ranks / 100.0).mean().item()

                    metrics_accum["HR10"].append(hit)
                    metrics_accum["NDCG"].append(ndcg)
                    metrics_accum["percentile"].append(percentile)

                val_loss = val_total / len(valid_loader)

                # aggregate
                mean_p = float(torch.tensor(metrics_accum["percentile"]).mean())
                mean_hr = float(torch.tensor(metrics_accum["HR10"]).mean())
                mean_ndcg = float(torch.tensor(metrics_accum["NDCG"]).mean())

            elapsed = (datetime.now() - start_time).total_seconds()

            # ---------------- LOGGING ---------------- #
            elapsed_min = int(elapsed // 60)
            elapsed_sec = elapsed % 60
            elapsed_str = f"{elapsed_min:02d}:{elapsed_sec:04.1f}"
            print(
                f"  {epoch:>3} |", 
                f" {avg_loss:.3f} |", 
                f" {val_loss:.3f} |", 
                f"{mean_p:.3f} |", 
                f"{mean_hr:.3f} |", 
                f"{mean_ndcg:.3f} |", 
                f"{cosθ} |", 
                f"    {elapsed_str}"
            )
    
    def save(self, path: str, note: str = None):
        """
        Save model checkpoint and optional note.
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "num_items": self.num_items,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dropout": self.dropout,
            "device": self.device,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": note
        }
        torch.save(checkpoint, path)

    def load(self, path: str, map_location="cpu"):
        """
        Load model checkpoint from file.
        The current instance must have the same architecture hyperparameters.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"])
        self.num_items = checkpoint.get("num_items", self.num_items)
        self.max_seq_len = checkpoint.get("max_seq_len", self.max_seq_len)
        self.device = checkpoint.get("device", self.device)
        print(f"Model loaded from {path}.")
        print(f"num_items:     {self.num_items}")
        print(f"max_seq_len:   {self.max_seq_len}")
        print(f"device:        {self.device}")
        print(f"batch_size:    {checkpoint.get('batch_size', None)}")
        print(f"lr:            {checkpoint.get('lr', None)}")
        print(f"weight_decay:  {checkpoint.get('weight_decay', None)}")
        print(f"num_epochs:    {checkpoint.get('num_epochs', None)}")
        print(f"saved_at:      {checkpoint.get('saved_at', None)}")
        print(f"note:          {checkpoint.get('note', None)}")