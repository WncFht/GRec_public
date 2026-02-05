import torch
from torch import nn


class ResKmeans(nn.Module):
    def __init__(
        self, n_layers, codebook_size, dim, extra_kmeans_config=None, **kwargs
    ):
        super().__init__()
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.extra_kmeans_config = extra_kmeans_config or {}
        self.centroids = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros((codebook_size, dim), requires_grad=False)
                )
                for i in range(n_layers)
            ]
        )

    def calc_loss(self, x, out, epsilon=1e-4):
        loss = ((out - x) ** 2).mean()
        rel_loss = (
            torch.abs(x - out)
            / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)
        ).mean()
        return {"loss": loss.item(), "rel_loss": rel_loss.item()}

    def train_kmeans(self, inputs, verbose=True):
        import time

        import faiss
        import numpy as np

        if isinstance(inputs, torch.Tensor):
            x0_t = inputs.detach()
            if x0_t.is_cuda:
                x0_t = x0_t.cpu()
            if x0_t.dtype != torch.float32:
                x0_t = x0_t.float()
            if not x0_t.is_contiguous():
                x0_t = x0_t.contiguous()
            x0 = x0_t.numpy()
        else:
            x0 = np.asarray(inputs, dtype=np.float32)
            if not x0.flags["C_CONTIGUOUS"]:
                x0 = np.ascontiguousarray(x0)

        if x0.ndim != 2 or int(x0.shape[1]) != int(self.dim):
            raise ValueError(
                f"Expected inputs shape [N, {self.dim}], got {tuple(x0.shape)}"
            )
        if int(x0.shape[0]) == 0:
            raise ValueError("Empty training data: inputs has 0 rows")

        # Residual training data (will be modified in-place)
        x = x0.copy()
        probe_idx = None
        if verbose and x.shape[0] > 0:
            probe_n = int(min(10_000, x.shape[0]))
            probe_idx = np.linspace(0, x.shape[0] - 1, probe_n, dtype=np.int64)
        for l in range(self.n_layers):
            t0 = time.time()
            if verbose:
                niter = self.extra_kmeans_config.get("niter", None)
                niter_str = str(niter) if niter is not None else "?"
                print(
                    f"[ResKmeans] layer {l}/{self.n_layers - 1} kmeans.train "
                    f"(n={x.shape[0]}, dim={self.dim}, k={self.codebook_size}, niter={niter_str})",
                    flush=True,
                )

            kmeans = faiss.Kmeans(
                self.dim,
                self.codebook_size,
                spherical=False,
                **self.extra_kmeans_config,
            )
            kmeans.train(x)
            if verbose:
                print(
                    f"[ResKmeans] layer {l} kmeans.train done in {time.time() - t0:.1f}s",
                    flush=True,
                )
            t1 = time.time()
            _, I = kmeans.index.search(x, 1)
            if verbose:
                print(
                    f"[ResKmeans] layer {l} assignment done in {time.time() - t1:.1f}s",
                    flush=True,
                )
            I = I.reshape([-1]).astype(np.int64, copy=False)
            t2 = time.time()

            centroids = kmeans.centroids.astype(np.float32, copy=False)
            chunk = 8192
            for start in range(0, x.shape[0], chunk):
                end = min(start + chunk, x.shape[0])
                x[start:end] -= centroids[I[start:end]]

            if verbose:
                probe_resid = x[probe_idx]
                flat = probe_resid.astype(np.float64, copy=False).ravel()
                mse = float(flat.dot(flat) / flat.size)
                print(f"{l} {{'loss': {mse:.6f}}}", flush=True)

            self.centroids[l] = nn.Parameter(
                torch.from_numpy(centroids.copy()), requires_grad=False
            )
            if verbose:
                print(
                    f"[ResKmeans] layer {l} residual update done in {time.time() - t2:.1f}s",
                    flush=True,
                )
            print(f"layer {l} finished", flush=True)

    def encode(self, x, n_layers=None):
        if n_layers is None:
            n_layers = self.n_layers
        else:
            assert n_layers <= self.n_layers
        out = []
        for l in range(n_layers):
            x_norm_sq = x.pow(2.0).sum(dim=1, keepdim=True)
            codebook_t_norm_sq = (
                self.centroids[l].T.pow(2.0).sum(dim=0, keepdim=True)
            )
            distances = torch.addmm(
                x_norm_sq + codebook_t_norm_sq,
                x,
                self.centroids[l].T,
                alpha=-2.0,
            )
            code = distances.argmin(dim=-1)
            x = x - self.centroids[l][code]
            out.append(code)
        out = torch.stack(out, dim=1)
        return out

    def decode(self, code):
        out = torch.zeros(
            (code.shape[0], self.dim), dtype=torch.float32, device=code.device
        )
        n_layers = code.shape[1]
        assert n_layers <= self.n_layers
        for l in range(n_layers):
            c = code[:, l]
            out += self.centroids[l][c]
        return out
