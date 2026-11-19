"""Subspace-Net 
Details
----------
Name: models.py
Authors: Dor Haim Shmuel
Created: 01/10/21
Edited: 02/06/23

Purpose:
--------
This script defines the tested NN-models and the model-based DL models, which used for simulation.
The implemented models:
    * DeepRootMUSIC: model-based deep learning algorithm as described in:
        [1] D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun and N. Shlezinger,
        "Deep Root Music Algorithm for Data-Driven Doa Estimation," ICASSP 2023 - 
        2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
        Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096504.
        
    * SubspaceNet: model-based deep learning algorithm as described in:
        [2] "SubspaceNet: Deep Learning-Aided Subspace methods for DoA Estimation".
    
    * DA-MUSIC: Deep Augmented MUSIC model-based deep learning algorithm as described in
        [3] J. P. Merkofer, G. Revach, N. Shlezinger, and R. J. van Sloun, “Deep
        augmented MUSIC algorithm for data-driven DoA estimation,” in IEEE
        International Conference on Acoustics, Speech and Signal Processing
        (ICASSP), 2022, pp. 3598-3602."
        
    * DeepCNN: Deep learning algorithm as described in:
        [4] G. K. Papageorgiou, M. Sellathurai, and Y. C. Eldar, “Deep networks
        for direction-of-arrival estimation in low SNR,” IEEE Trans. Signal
        Process., vol. 69, pp. 3714-3729, 2021.

Functions:
----------
This script also includes the implementation of Root-MUSIC algorithm, as it is written using Pytorch library,
for the usage of src.models: SubspaceNet implementation.
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import warnings
from src.utils import gram_diagonal_overload, device
from src.utils import sum_of_diags_torch, find_roots_torch


warnings.simplefilter("ignore")
# Constants
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelGenerator(object):
    """
    Generates an instance of the desired model, according to model configuration parameters.
    """

    def __init__(self):
        """
        Initialize ModelParams object.
        """
        self.model_type = None
        self.diff_method = None
        self.tau = None
        self.variant = {}  # constructor overrides for the module ablations (SubspaceNetAttn / SubspaceNetIter)

    def set_tau(self, tau: int = None):
        """
        Set the value of tau parameter for SubspaceNet model.

        Parameters:
            tau (int): The number of lags.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If tau parameter is not provided for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if not isinstance(tau, int):
                raise ValueError(
                    "ModelParams.set_tau: tau parameter must be provided for SubspaceNet model"
                )
            self.tau = tau
        return self

    def set_diff_method(self, diff_method: str = "root_music"):
        """
        Set the differentiation method for SubspaceNet model.

        Parameters:
            diff_method (str): The differantiable subspace method ("esprit" or "root_music").

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If the diff_method is not defined for SubspaceNet model.
        """
        if self.model_type.startswith("SubspaceNet"):
            if diff_method not in ["esprit", "root_music"]:
                raise ValueError(
                    f"ModelParams.set_diff_method: {diff_method} is not defined for SubspaceNet model"
                )
            self.diff_method = diff_method
        return self

    def set_model_type(self, model_type: str):
        """
        Set the model type.

        Parameters:
            model_type (str): The model type.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            ValueError: If model type is not provided.
        """
        if not isinstance(model_type, str):
            raise ValueError(
                "ModelParams.set_model_type: model type has not been provided"
            )
        self.model_type = model_type
        return self

    def set_model(self, system_model_params):
        """
        Set the model based on the model type and system model parameters.

        Parameters:
            system_model_params (SystemModelParams): The system model parameters.

        Returns:
            ModelParams: The updated ModelParams object.

        Raises:
            Exception: If the model type is not defined.
        """
        if self.model_type.startswith("DA-MUSIC"):
            self.model = DeepAugmentedMUSIC(
                N=system_model_params.N,
                T=system_model_params.T,
                M=system_model_params.M,
            )
        elif self.model_type.startswith("DeepCNN"):
            self.model = DeepCNN(N=system_model_params.N, grid_size=361)
        elif self.model_type.startswith("SubspaceNetAttn"):
            # NOTE: must be checked before the generic "SubspaceNet" branch,
            # since "SubspaceNetAttn".startswith("SubspaceNet") is also True.
            self.model = SubspaceNetAttn(
                tau=self.tau,
                N=system_model_params.N,
                M=system_model_params.M,
                diff_method=self.diff_method,
                **self.variant,
            )
        elif self.model_type.startswith("SubspaceNetIter"):
            # NOTE: must be checked before the generic "SubspaceNet" branch,
            # since "SubspaceNetIter".startswith("SubspaceNet") is also True.
            self.model = SubspaceNetIter(
                tau=self.tau,
                N=system_model_params.N,
                M=system_model_params.M,
                diff_method=self.diff_method,
                **self.variant,
            )
        elif self.model_type.startswith("SubspaceNet"):
            self.model = SubspaceNet(
                tau=self.tau, M=system_model_params.M, diff_method=self.diff_method
            )
        else:
            raise Exception(
                f"ModelGenerator.set_model: Model type {self.model_type} is not defined"
            )
        return self


class DeepRootMUSIC(nn.Module):
    """DeepRootMUSIC is model-based deep learning model for DOA estimation problem.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        LeakyReLU (nn.LeakyReLU): Leaky reLu activation function, with activation_value.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, activation_value: float):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            activation_value (float): Value for the activation function.

        """
        super(DeepRootMUSIC, self).__init__()
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=2)
        self.LeakyReLU = nn.LeakyReLU(activation_value)
        self.DropOut = nn.Dropout(0.2)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the DeepRootMUSIC.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.LeakyReLU(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.LeakyReLU(x)
        # DCNN block #1
        x = self.deconv1(x)
        x = self.LeakyReLU(x)
        # DCNN block #2
        x = self.deconv2(x)
        x = self.LeakyReLU(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv3(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(Kx_tag, eps=1)  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Root-MUSIC algorithm
        doa_prediction, doa_all_predictions, roots = root_music(
            Rz, self.M, self.batch_size
        )
        return doa_prediction, doa_all_predictions, roots, Rz


# TODO: inherit SubspaceNet from DeepRootMUSIC
class SubspaceNet(nn.Module):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        conv1 (nn.Conv2d): Convolution layer 1.
        conv2 (nn.Conv2d): Convolution layer 2.
        conv3 (nn.Conv2d): Convolution layer 3.
        deconv1 (nn.ConvTranspose2d): De-convolution layer 1.
        deconv2 (nn.ConvTranspose2d): De-convolution layer 2.
        deconv3 (nn.ConvTranspose2d): De-convolution layer 3.
        DropOut (nn.Dropout): Dropout layer.
        ReLU (nn.ReLU): ReLU activation function.

    Methods:
    --------
        anti_rectifier(X): Applies the anti-rectifier operation to the input tensor.
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.
        gram_diagonal_overload(Kx, eps): Applies Gram operation and diagonal loading to a complex matrix.

    """

    def __init__(self, tau: int, M: int, diff_method: str = "root_music"):
        """Initializes the SubspaceNet model.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            M (int): Number of sources.

        """
        super(SubspaceNet, self).__init__()
        self.M = M
        self.tau = tau
        self.conv1 = nn.Conv2d(self.tau, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=2)
        self.deconv3 = nn.ConvTranspose2d(64, 16, kernel_size=2)
        self.deconv4 = nn.ConvTranspose2d(32, 1, kernel_size=2)
        self.DropOut = nn.Dropout(0.2)
        self.ReLU = nn.ReLU()
        # Set the subspace method for training
        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training subspaceNet.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNet
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNet.set_diff_method: Method {diff_method} is not defined for SubspaceNet"
            )

    def anti_rectifier(self, X):
        """Applies the anti-rectifier operation to the input tensor.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: Output tensor after applying the anti-rectifier operation.

        """
        return torch.cat((self.ReLU(X), self.ReLU(-X)), 1)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to the differentiable subspace algorithm
        method_output = self.diff_method(Rz, self.M, self.batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz


class SubspaceNetAttn(nn.Module):
    """SubspaceNetAttn is an original variant of SubspaceNet that replaces the
    stacked-convolution encoder with a Transformer-style self-attention
    mechanism across the tau auto-correlation lags.

    Motivation
    ----------
    In the original SubspaceNet, the tau lag matrices are stacked as input
    *channels* to a 2D CNN (`nn.Conv2d(self.tau, 16, kernel_size=2)`). This
    means the network can only combine lags through local, fixed convolution
    kernels stacked over several layers -- there is no explicit mechanism for
    one lag to directly attend to a distant, non-adjacent lag.

    SubspaceNetAttn instead treats each lag matrix Rx_tau[:, k] as a single
    token, embeds it, and lets all tau tokens attend to each other via
    multi-head self-attention before pooling them into a single context
    vector that is decoded into the surrogate covariance matrix. This allows
    the network to learn which combinations of lags (not necessarily
    adjacent ones) are most informative for a given scenario (e.g. coherent
    sources, low SNR, few snapshots).

    The output signature is identical to SubspaceNet's
    (doa_prediction, doa_all_predictions, roots, Rz), so it plugs directly
    into the existing training/evaluation pipeline (any code that checks
    `model_type.startswith("SubspaceNet")` also matches "SubspaceNetAttn").

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        N (int): Number of array sensors.
        lag_embed (nn.Sequential): Per-lag embedding MLP.
        pos_embedding (nn.Parameter): Learnable positional embedding per lag.
        transformer (nn.TransformerEncoder): Self-attention across lags.
        pool_attn (nn.MultiheadAttention): Learned attention-pooling over lags.
        recon_head (nn.Sequential): MLP decoder reconstructing the NxN
            surrogate covariance (real & imaginary parts).

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of SubspaceNetAttn.

    """

    def __init__(
        self,
        tau: int,
        N: int,
        M: int,
        diff_method: str = "esprit",
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_pos: bool = True,
        pool: str = "attn",
        output: str = "full",
    ):
        """Initializes the SubspaceNetAttn model.

        Ablation switches (defaults = the full model): num_layers=0 removes the
        Transformer layers, use_pos=False the positional embedding, pool="mean"
        replaces the attention pooling by a mean over the lag tokens.
        output: "full" predicts the whole surrogate covariance K^H K + I; "res_psd" and
        "res_herm" add a correction to the lag-0 sample covariance instead (X6, reviewer 2, M2):
        the same positive-definite K^H K + I, or the Hermitian part of K.

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            N (int): Number of array sensors.
            M (int): Number of sources.
            diff_method (str): differentiable subspace method, "root_music" or "esprit".
            d_model (int): embedding dimension used by the attention block.
            nhead (int): number of attention heads.
            num_layers (int): number of stacked Transformer encoder layers.
            dropout (float): dropout probability.
        """
        super(SubspaceNetAttn, self).__init__()
        self.M = M
        self.tau = tau
        self.N = N
        self.d_model = d_model

        # Per-lag embedding: each lag matrix Rx_tau[:, k] has shape [2N, N],
        # flattened and projected into a d_model-dim token.
        input_dim = 2 * N * N
        self.lag_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        # Learnable positional embedding: lag order carries physical meaning
        # (larger tau = larger delay), so unlike a typical NLP transformer we
        # don't want a permutation-invariant treatment of lags.
        self.pos_embedding = nn.Parameter(torch.randn(1, tau, d_model) * 0.02) if use_pos else None

        if num_layers:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.transformer = nn.Identity()

        # Attention pooling: a learned query attends over the tau contextualized
        # lag tokens to produce a single summary vector (instead of e.g. a
        # naive mean, letting the network weigh informative lags more).
        self.pool = pool
        assert output in ("full", "res_psd", "res_herm"), output
        self.output = output
        if pool == "attn":
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, batch_first=True
            )

        # Reconstruction head: decodes the pooled context vector into the
        # real & imaginary parts of the NxN surrogate covariance matrix.
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, 2 * N * N),
        )

        # Set the subspace method for training (reuses SubspaceNet's logic)
        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training SubspaceNetAttn.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNetAttn
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNetAttn.set_diff_method: Method {diff_method} is not defined for SubspaceNetAttn"
            )

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of SubspaceNetAttn.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        batch_size = Rx_tau.shape[0]
        # Flatten each lag matrix into a token: [B, tau, 2N*N]
        tokens = Rx_tau.reshape(batch_size, self.tau, -1)
        tokens = self.lag_embed(tokens)  # [B, tau, d_model]
        if self.pos_embedding is not None:
            tokens = tokens + self.pos_embedding
        # Self-attention across all tau lags jointly (global, non-local mixing)
        tokens = self.transformer(tokens)  # [B, tau, d_model]
        # Attention-pool the tau tokens into a single context vector
        if self.pool == "attn":
            query = self.pool_query.expand(batch_size, -1, -1)  # [B, 1, d_model]
            pooled, _ = self.pool_attn(query, tokens, tokens)  # [B, 1, d_model]
            pooled = pooled.squeeze(1)  # [B, d_model]
        else:
            pooled = tokens.mean(dim=1)
        # Decode into the surrogate covariance matrix
        out = self.recon_head(pooled)  # [B, 2*N*N]
        out = out.view(batch_size, 2, self.N, self.N)
        Rx_real = out[:, 0]  # [B, N, N]
        Rx_imag = out[:, 1]  # [B, N, N]
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # [B, N, N]
        if self.output == "res_herm":
            Rz = (Kx_tag + Kx_tag.conj().transpose(1, 2)) / 2
        else:
            # Apply Gram operation + diagonal loading -> valid Hermitian PSD matrix
            Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=batch_size)
        if self.output != "full":
            # Residual output: the network output corrects the lag-0 sample covariance
            R0 = torch.complex(Rx_tau[:, 0, : self.N], Rx_tau[:, 0, self.N :])
            Rz = Rz + (R0 + R0.conj().transpose(1, 2)) / 2
        # Feed surrogate covariance to the differentiable subspace algorithm
        method_output = self.diff_method(Rz, self.M, batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz


class SubspaceNetIter(nn.Module):
    """SubspaceNetIter is an original variant of SubspaceNet based on
    *deep unfolding* / recurrent iterative refinement, rather than a
    single-shot encoder-decoder (SubspaceNet) or attention-based encoder
    (SubspaceNetAttn).

    Motivation
    ----------
    Every existing surrogate-covariance architecture for this problem --
    the original SubspaceNet autoencoder, attention-based variants such as
    SubspaceNetAttn, and other related-work designs (e.g. CNN-only, RNN
    architectures like DA-MUSIC, Toeplitz-prior autoencoders) -- computes
    the surrogate covariance in a *single forward pass*: raw features in,
    final covariance out. None of them iteratively refine an initial
    estimate over multiple steps.

    SubspaceNetIter instead borrows the "deep unfolding" idea, well
    established in other inverse-problem settings (e.g. compressed
    sensing, image restoration), but not previously applied to the
    surrogate-covariance estimation problem for DoA. It:
      1. Embeds each of the tau lag matrices individually (full detail
         preserved per lag, as in SubspaceNetAttn).
      2. Produces an initial coarse covariance estimate from a simple
         mean-pooled summary of the lag tokens.
      3. Refines this estimate over `n_iterations` steps using a *shared*
         (weight-tied) recurrent cell. Critically, at every step the
         module *re-attends* over the lag tokens using the current
         covariance estimate as a query (cross-attention), rather than
         reusing one fixed context vector computed once up front --
         i.e. each refinement step can focus on different lag evidence
         depending on what the current estimate still gets wrong. This
         combines SubspaceNetAttn's cross-lag mixing with SubspaceNetIter's
         iterative-refinement idea in a way neither original architecture
         does on its own: SubspaceNetAttn attends over lags exactly once
         before pooling, and the *first* version of SubspaceNetIter used a
         single fixed context vector reused unchanged at every refinement
         step (found, by direct comparison, to leave a sizeable performance
         gap versus SubspaceNetAttn -- see accompanying analysis).
      4. Reconstructs the surrogate covariance from the final state.

    The output signature matches SubspaceNet's
    (doa_prediction, doa_all_predictions, roots, Rz), so it plugs directly
    into the existing training/evaluation pipeline.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.
        N (int): Number of array sensors.
        n_iterations (int): Number of shared-weight refinement steps.
        lag_embed (nn.Sequential): Per-lag embedding MLP (full detail per lag).
        query_proj (nn.Linear): Projects the current covariance state into
            a query vector used to re-attend over the lag tokens.
        cross_attn (nn.MultiheadAttention): Query-conditioned cross-attention
            over the lag tokens, applied fresh at every refinement step.
        init_head (nn.Linear): Produces the initial coarse covariance estimate.
        refine_cell (nn.GRUCell): Shared recurrent refinement cell, applied
            `n_iterations` times (deep-unfolding: weight-tied across steps).

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of SubspaceNetIter.

    """

    def __init__(
        self,
        tau: int,
        N: int,
        M: int,
        diff_method: str = "esprit",
        n_iterations: int = 5,
        d_ctx: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        context: str = "attn",
    ):
        """Initializes the SubspaceNetIter model.

        Ablation switches (defaults = the full model): n_iterations=0 removes the
        refinement loop, context="mean" replaces the cross-attention by the mean
        of the lag tokens (same context at every step).

        Args:
        -----
            tau (int): Number of auto-correlation lags.
            N (int): Number of array sensors.
            M (int): Number of sources.
            diff_method (str): differentiable subspace method, "root_music" or "esprit".
            n_iterations (int): number of shared-weight refinement steps.
            d_ctx (int): dimension of the evidence context vector / attention embedding.
            n_heads (int): number of attention heads for the iterative cross-attention.
            dropout (float): dropout probability in the reconstruction head.
        """
        super(SubspaceNetIter, self).__init__()
        self.M = M
        self.tau = tau
        self.N = N
        self.n_iterations = n_iterations
        self.state_dim = 2 * N * N  # flattened [real;imag] NxN covariance

        # Per-lag embedding: each flattened lag matrix Rx_tau[:, k] (shape
        # [2N, N]) is projected to a d_ctx-dim token, preserving full detail
        # per lag (as in SubspaceNetAttn) rather than compressing all lags
        # together immediately.
        input_dim = 2 * N * N
        self.lag_embed = nn.Sequential(
            nn.Linear(input_dim, d_ctx),
            nn.LayerNorm(d_ctx),
            nn.GELU(),
        )

        # Query-conditioned cross-attention: at every refinement step, the
        # *current* covariance estimate is projected into a query that
        # attends over the (fixed) lag tokens, so different steps can pull
        # different evidence from the lags as the estimate evolves.
        self.context = context
        if n_iterations and context == "attn":
            self.query_proj = nn.Linear(self.state_dim, d_ctx)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_ctx, num_heads=n_heads, batch_first=True
            )

        # Initial coarse covariance estimate from a simple mean-pooled
        # summary of the lag tokens (the iterative cross-attention refines
        # this from here on).
        self.init_head = nn.Linear(d_ctx, self.state_dim)

        # Shared (weight-tied) recurrent refinement cell: repeatedly maps
        # (re-attended context, current covariance estimate) -> updated
        # covariance estimate. This is the deep-unfolding step.
        if n_iterations:
            self.refine_cell = nn.GRUCell(input_size=d_ctx, hidden_size=self.state_dim)

        # Dedicated reconstruction head applied to the final refined state,
        # matching SubspaceNetAttn's decoder capacity (previously the GRU
        # hidden state was reshaped directly into the covariance with no
        # further non-linear transformation -- this gave the model less
        # capacity to finalize/clean up the estimate than SubspaceNetAttn's
        # 2-layer, 4x-expansion decoder).
        self.recon_head = nn.Sequential(
            nn.Linear(self.state_dim, 4 * d_ctx),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_ctx, self.state_dim),
        )

        self.set_diff_method(diff_method)

    def set_diff_method(self, diff_method: str):
        """Sets the differentiable subspace method for training SubspaceNetIter.
            Options: "root_music", "esprit"

        Args:
        -----
            diff_method (str): differentiable subspace method.

        Raises:
        -------
            Exception: Method diff_method is not defined for SubspaceNetIter
        """
        if diff_method.startswith("root_music"):
            self.diff_method = root_music
        elif diff_method.startswith("esprit"):
            self.diff_method = esprit
        else:
            raise Exception(
                f"SubspaceNetIter.set_diff_method: Method {diff_method} is not defined for SubspaceNetIter"
            )

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of SubspaceNetIter.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            doa_all_predictions (torch.Tensor): All DOA predictions for each root, over all batches.
            roots_to_return (torch.Tensor): The unsorted roots.
            Rz (torch.Tensor): Surrogate covariance matrix (after the final refinement step).

        """
        batch_size = Rx_tau.shape[0]

        # Per-lag embedding (full detail per lag, no early compression)
        tokens = Rx_tau.reshape(batch_size, self.tau, -1)  # [B, tau, 2N*N]
        tokens = self.lag_embed(tokens)  # [B, tau, d_ctx]

        # Initial coarse estimate from a simple mean-pooled summary
        state = self.init_head(tokens.mean(dim=1))  # [B, state_dim]

        # Iterative refinement: the query re-attends over the lag tokens
        # at every step, conditioned on the current state.
        for _ in range(self.n_iterations):
            if self.context == "attn":
                query = self.query_proj(state).unsqueeze(1)  # [B, 1, d_ctx]
                context, _ = self.cross_attn(query, tokens, tokens)  # [B, 1, d_ctx]
                context = context.squeeze(1)  # [B, d_ctx]
            else:
                context = tokens.mean(dim=1)
            state = self.refine_cell(context, state)

        # Final reconstruction: a dedicated decoder cleans up/finalizes the
        # refined state into the covariance representation, instead of
        # reshaping the raw GRU hidden state directly.
        out = self.recon_head(state).view(batch_size, 2, self.N, self.N)
        Rx_real = out[:, 0]
        Rx_imag = out[:, 1]
        Kx_tag = torch.complex(Rx_real, Rx_imag)
        Rz = gram_diagonal_overload(Kx=Kx_tag, eps=1, batch_size=batch_size)
        method_output = self.diff_method(Rz, self.M, batch_size)
        if isinstance(method_output, tuple):
            # Root MUSIC output
            doa_prediction, doa_all_predictions, roots = method_output
        else:
            # Esprit output
            doa_prediction = method_output
            doa_all_predictions, roots = None, None
        return doa_prediction, doa_all_predictions, roots, Rz


class SubspaceNetEsprit(SubspaceNet):
    """SubspaceNet is model-based deep learning model for generalizing DOA estimation problem,
        over subspace methods.
        SubspaceNetEsprit is based on the ability to perform back-propagation using ESPRIT algorithm,
        instead of RootMUSIC.

    Attributes:
    -----------
        M (int): Number of sources.
        tau (int): Number of auto-correlation lags.

    Methods:
    --------
        forward(Rx_tau): Performs the forward pass of the SubspaceNet.

    """

    def __init__(self, tau: int, M: int):
        super().__init__(tau, M)

    def forward(self, Rx_tau: torch.Tensor):
        """
        Performs the forward pass of the SubspaceNet.

        Args:
        -----
            Rx_tau (torch.Tensor): Input tensor of shape [Batch size, tau, 2N, N].

        Returns:
        --------
            doa_prediction (torch.Tensor): The predicted direction-of-arrival (DOA) for each batch sample.
            Rz (torch.Tensor): Surrogate covariance matrix.

        """
        # Rx_tau shape: [Batch size, tau, 2N, N]
        self.N = Rx_tau.shape[-1]
        self.batch_size = Rx_tau.shape[0]
        ## Architecture flow ##
        # CNN block #1
        x = self.conv1(Rx_tau)
        x = self.anti_rectifier(x)
        # CNN block #2
        x = self.conv2(x)
        x = self.anti_rectifier(x)
        # CNN block #3
        x = self.conv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #1
        x = self.deconv2(x)
        x = self.anti_rectifier(x)
        # DCNN block #2
        x = self.deconv3(x)
        x = self.anti_rectifier(x)
        # DCNN block #3
        x = self.DropOut(x)
        Rx = self.deconv4(x)
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_View = Rx.view(Rx.size(0), Rx.size(2), Rx.size(3))
        # Real and Imaginary Reconstruction
        Rx_real = Rx_View[:, : self.N, :]  # Shape: [Batch size, N, N])
        Rx_imag = Rx_View[:, self.N :, :]  # Shape: [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape: [Batch size, N, N])
        # Apply Gram operation diagonal loading
        Rz = gram_diagonal_overload(
            Kx=Kx_tag, eps=1, batch_size=self.batch_size
        )  # Shape: [Batch size, N, N]
        # Feed surrogate covariance to Esprit algorithm
        doa_prediction = esprit(Rz, self.M, self.batch_size)
        return doa_prediction, Rz


class DeepAugmentedMUSIC(nn.Module):
    """DeepAugmentedMUSIC is a model-based deep learning model for Direction of Arrival (DOA) estimation.

    Attributes:
        N (int): Number of sensors.
        T (int): Number of observations.
        M (int): Number of sources.
        angels (torch.Tensor): Tensor containing angles from -pi/2 to pi/2.
        input_size (int): Size of the input.
        hidden_size (int): Size of the hidden layer.
        rnn (nn.GRU): Recurrent neural network module.
        fc (nn.Linear): Fully connected layer.
        fc1 (nn.Linear): Fully connected layer.
        fc2 (nn.Linear): Fully connected layer.
        fc3 (nn.Linear): Fully connected layer.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.
        DropOut (nn.Dropout): Dropout layer.
        BatchNorm (nn.BatchNorm1d): Batch normalization layer.
        sv (torch.Tensor): Steering vector.

    Methods:
        steering_vec(): Computes the steering vector based on the specified parameters.
        spectrum_calculation(Un: torch.Tensor): Calculates the MUSIC spectrum.
        pre_MUSIC(Rz: torch.Tensor): Applies the MUSIC operation for generating the spectrum.
        forward(X: torch.Tensor): Performs the forward pass of the DeepAugmentedMUSIC model.
    """

    def __init__(self, N: int, T: int, M: int):
        """Initializes the DeepAugmentedMUSIC model.

        Args:
        -----
            N (int): Number of sensors.
            M (int): Number of sources.
            T (int): Number of observations.
        """
        super(DeepAugmentedMUSIC, self).__init__()
        self.N, self.T, self.M = N, T, M
        self.angels = torch.linspace(-1 * np.pi / 2, np.pi / 2, 361)
        self.input_size = 2 * self.N
        self.hidden_size = 2 * self.N
        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.N)
        self.fc1 = nn.Linear(self.angels.shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.M)
        self.ReLU = nn.ReLU()
        self.DropOut = nn.Dropout(0.25)
        self.BatchNorm = nn.BatchNorm1d(self.T)
        self.sv = self.steering_vec()
        # Weights initialization
        nn.init.xavier_uniform(self.fc.weight)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.xavier_uniform(self.fc3.weight)

    def steering_vec(self):
        """Computes the ideal steering vector based on the specified parameters.
            equivalent to src.system_model.steering_vec method, but support pyTorch.

        Returns:
        --------
            tensor.Torch: the steering vector
        """
        sv = []
        for angle in self.angels:
            a = torch.exp(
                -1 * 1j * np.pi * torch.linspace(0, self.N - 1, self.N) * np.sin(angle)
            )
            sv.append(a)
        return torch.stack(sv, dim=0)

    def spectrum_calculation(self, Un: torch.Tensor):
        """Vectorized MUSIC spectrum calculation across all candidate angles.

        Computes, for every steering vector sv_i in self.sv:
            spectrum_equation[i] = real( conj(sv_i) @ Un @ Un^H @ sv_i )
        for a single sample (Un: [N, N-M]), or batched over samples
        (Un: [B, N, N-M]) -- in both cases without any Python-level loop
        over the 361 angles (this used to be a sequential for-loop, which
        was the dominant training-time bottleneck for this model).
        """
        SV = self.sv.to(Un.device)  # [361, N]
        if Un.dim() == 2:
            # Single sample: P = Un @ Un^H, shape [N, N]
            P = Un @ torch.conj(Un).T
            spectrum_equation = torch.einsum(
                "ij,jk,ik->i", torch.conj(SV), P, SV
            ).real
        else:
            # Batched: Un shape [B, N, N-M] -> P shape [B, N, N]
            P = Un @ torch.conj(Un).transpose(-2, -1)
            spectrum_equation = torch.einsum(
                "ij,bjk,ik->bi", torch.conj(SV), P, SV
            ).real
        spectrum = 1 / spectrum_equation
        return spectrum, spectrum_equation

    def pre_MUSIC(self, Rz: torch.Tensor):
        """Applies the MUSIC operation for generating the spectrum.

        Vectorized over the batch dimension (was previously a Python
        for-loop over every sample in the batch, each doing its own
        eigendecomposition + a further 361-angle inner loop -- with
        batch_size=2048 this meant ~740K sequential small ops per
        training batch. torch.linalg.eig natively supports a batched
        input, so this is now a single batched call.)

        Args:
            Rz (torch.Tensor): Generated covariance matrix, shape [B, N, N]

        Returns:
            torch.Tensor: The generated MUSIC spectrum, shape [B, 361]
        """
        # Batched eigendecomposition: eigenvectors shape [B, N, N]
        eigenvalues, eigenvectors = torch.linalg.eig(Rz)
        # Sort by |eigenvalue|, largest first (eig returns them unordered)
        order = torch.argsort(torch.abs(eigenvalues), dim=-1, descending=True)
        eigenvectors = torch.gather(
            eigenvectors, 2, order.unsqueeze(1).expand(-1, Rz.shape[1], -1)
        )
        # Noise subspace: eigenvectors associated with the smallest N-M eigenvalues
        Un = eigenvectors[:, :, self.M :]
        spectrum, _ = self.spectrum_calculation(Un)
        return spectrum

    def forward(self, X: torch.Tensor):
        """
        Performs the forward pass of the DeepAugmentedMUSIC model.

        Args:
        -----
            X (torch.Tensor): Input tensor.

        Returns:
        --------
            torch.Tensor: The estimated DOA.
        """
        # X shape == [Batch size, N, T]
        self.BATCH_SIZE = X.shape[0]
        ## Architecture flow ##
        # decompose X and concatenate real and imaginary part
        X = torch.cat(
            (torch.real(X), torch.imag(X)), 1
        )  # Shape ==  [Batch size, 2N, T]
        # Reorder to [Batch size, T, 2N] (permute, not view: view scrambles sensors and time)
        X = X.permute(0, 2, 1)
        # Apply batch normalization
        X = self.BatchNorm(X)
        # GRU Clock
        gru_out, hn = self.rnn(X)
        Rx = gru_out[:, -1]
        # Reshape Output shape: [Batch size, 1, 2N]
        Rx = Rx.view(Rx.size(0), 1, Rx.size(1))
        # FC Block
        Rx = self.fc(Rx)  # Shape: [Batch size, 1, 2N^2])
        # Reshape Output shape: [Batch size, 2N, N]
        Rx_view = Rx.view(self.BATCH_SIZE, 2 * self.N, self.N)
        Rx_real = Rx_view[:, : self.N, :]  # Shape [Batch size, N, N])
        Rx_imag = Rx_view[:, self.N :, :]  # Shape [Batch size, N, N])
        Kx_tag = torch.complex(Rx_real, Rx_imag)  # Shape [Batch size, N, N])
        # Build MUSIC spectrum
        spectrum = self.pre_MUSIC(Kx_tag)  # Shape [Batch size, 361(grid_size)])
        # Apply peak detection using FC block #2
        y = self.ReLU(self.fc1(spectrum))  # Shape [Batch size, 361(grid_size)])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N])
        y = self.ReLU(self.fc2(y))  # Shape [Batch size, 2N)
        # Find doa
        DOA = self.fc3(y)  # Shape [Batch size, M)
        return DOA


class DeepCNN(nn.Module):
    """DeepCNN is a convolutional neural network model for DoA  estimation.

    Args:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.

    Attributes:
        N (int): Input dimension size.
        grid_size (int): Size of the output grid.
        conv1 (nn.Conv2d): Convolutional layer 1.
        conv2 (nn.Conv2d): Convolutional layer 2.
        fc1 (nn.Linear): Fully connected layer 1.
        BatchNorm (nn.BatchNorm2d): Batch normalization layer.
        fc2 (nn.Linear): Fully connected layer 2.
        fc3 (nn.Linear): Fully connected layer 3.
        fc4 (nn.Linear): Fully connected layer 4.
        DropOut (nn.Dropout): Dropout layer.
        Sigmoid (nn.Sigmoid): Sigmoid activation function.
        ReLU (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
        forward(X: torch.Tensor): Performs the forward pass of the DeepCNN model.
    """

    def __init__(self, N, grid_size):
        ## input dim (N, T)
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size
        self.conv1 = nn.Conv2d(3, 256, kernel_size=3)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2)
        self.fc1 = nn.Linear(256 * (self.N - 5) * (self.N - 5), 4096)
        self.BatchNorm = nn.BatchNorm2d(256)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)
        self.DropOut = nn.Dropout(0.3)
        self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, X):
        # X shape == [Batch size, N, N, 3]
        # [Batch size, 3, N, N] (permute, not view: view mixes channels); contiguous so the
        # later flatten (.view) works
        X = X.permute(0, 3, 1, 2).contiguous()
        ## Architecture flow ##
        # CNN block #1: 3xNxN-->256x(N-2)x(N-2)
        X = self.conv1(X)
        X = self.ReLU(X)
        # CNN block #2: 256x(N-2)x(N-2)-->256x(N-3)x(N-3)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #3: 256x(N-3)x(N-3)-->256x(N-4)x(N-4)
        X = self.conv2(X)
        X = self.ReLU(X)
        # CNN block #4: 256x(N-4)x(N-4)-->256x(N-5)x(N-5)
        X = self.conv2(X)
        X = self.ReLU(X)
        # FC BLOCK
        # Reshape Output shape: [Batch size, 256 * (self.N - 5) * (self.N - 5)]
        X = X.view(X.size(0), -1)
        X = self.DropOut(self.ReLU(self.fc1(X)))  # [Batch size, 4096]
        X = self.DropOut(self.ReLU(self.fc2(X)))  # [Batch size, 2048]
        X = self.DropOut(self.ReLU(self.fc3(X)))  # [Batch size, 1024]
        X = self.fc4(X)  # [Batch size, grid_size]
        X = self.Sigmoid(X)
        return X


def root_music(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Root-MUSIC algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent tosrc.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
        doa_all_batches (torch.Tensor): All doa predicted, given all roots, over all batches.
        roots_to_return (torch.Tensor): The unsorted roots.
    """
    # Batched over samples (same result as the former per-sample loop; batch_size unused)
    dist, f = 0.5, 1
    N = Rz.shape[-1]
    eigenvalues, eigenvectors = torch.linalg.eig(Rz)
    # Noise subspace: eigenvectors of the N - M smallest |eigenvalues|
    order = torch.argsort(torch.abs(eigenvalues), dim=-1, descending=True)[:, M:]
    Un = torch.gather(eigenvectors, 2, order.unsqueeze(1).expand(-1, N, -1))
    F = Un @ Un.conj().transpose(1, 2)
    # Polynomial coefficients: sums of the diagonals of F, offsets -(N-1) .. N-1
    coeffs = torch.stack([torch.diagonal(F, k, dim1=1, dim2=2).sum(-1) for k in range(-N + 1, N)], dim=1)
    # Roots = eigenvalues of the companion matrix
    A = torch.diag(torch.ones(2 * N - 3, dtype=coeffs.dtype, device=coeffs.device), -1).repeat(len(Rz), 1, 1)
    A[:, 0, :] = -coeffs[:, 1:] / coeffs[:, :1]
    roots = torch.linalg.eigvals(A)
    to_doa = lambda r: torch.arcsin((1 / (2 * np.pi * dist * f)) * torch.angle(r))
    # The M roots closest to the unit circle among those inside it
    gap = torch.abs(torch.abs(roots) - 1).masked_fill(torch.abs(roots) >= 1, float("inf"))
    closest = torch.argsort(gap, dim=-1)[:, :M]
    # Third output as before: the unsorted roots of the last sample
    return to_doa(torch.gather(roots, 1, closest)), to_doa(roots), roots[-1]


def esprit(Rz: torch.Tensor, M: int, batch_size: int):
    """Implementation of the model-based Esprit algorithm, support Pytorch, intended for
        MB-DL models. the model sets for nominal and ideal condition (Narrow-band, ULA, non-coherent)
        as it accepts the surrogate covariance matrix.
        it is equivalent to src.methods: RootMUSIC.narrowband() method.

    Args:
    -----
        Rz (torch.Tensor): Focused covariance matrix
        M (int): Number of sources
        batch_size: the number of batches

    Returns:
    --------
        doa_batches (torch.Tensor): The predicted doa, over all batches.
    """

    # Batched over samples (same result as the former per-sample loop; batch_size unused)
    # Extract eigenvalues and eigenvectors using EVD
    eigenvalues, eigenvectors = torch.linalg.eig(Rz)
    # Signal subspace: eigenvectors of the M largest |eigenvalues|
    order = torch.argsort(torch.abs(eigenvalues), dim=-1, descending=True)[:, :M]
    Us = torch.gather(eigenvectors, 2, order.unsqueeze(1).expand(-1, Rz.shape[1], -1))
    # Rotational invariance between the two overlapping sub-arrays
    phi = torch.linalg.pinv(Us[:, :-1]) @ Us[:, 1:]
    phi_eigenvalues, _ = torch.linalg.eig(phi)
    # Calculate the DoA out of the phase component
    return -1 * torch.arcsin((1 / np.pi) * torch.angle(phi_eigenvalues))
