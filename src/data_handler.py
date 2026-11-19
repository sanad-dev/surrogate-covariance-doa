"""Subspace-Net 
Details
----------
    Name: data_handler.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 03/06/23

Purpose:
--------
    This scripts handle the creation and processing of synthetic datasets
    based on specified parameters and model types.
    It includes functions for generating datasets, reading data from files,
    computing autocorrelation matrices, and creating covariance tensors.

Attributes:
-----------
    Samples (from src.signal_creation): A class for creating samples used in dataset generation.

    The script defines the following functions:
    * create_dataset: Generates a synthetic dataset based on the specified parameters and model type.
    * read_data: Reads data from a file specified by the given path.
    * autocorrelation_matrix: Computes the autocorrelation matrix for a given lag of the input samples.
    * create_autocorrelation_tensor: Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.
    * create_cov_tensor: Creates a 3D tensor containing the real part,
        imaginary part, and phase component of the covariance matrix.
    * set_dataset_filename: Returns the generic suffix of the datasets filename.

"""

# Imports
import torch
import numpy as np
import itertools
import math
import random
from tqdm import tqdm
from src.signal_creation import Samples
from pathlib import Path
from src.system_model import SystemModelParams

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_dataset(
    system_model_params: SystemModelParams,
    samples_size: float,
    model_type: str,
    tau: int = None,
    save_datasets: bool = False,
    datasets_path: Path = None,
    true_doa: list = None,
    phase: str = None,
):
    """
    Generates a synthetic dataset based on the specified parameters and model type.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams
        samples_size (float): The size of the dataset.
        tau (int): The number of lags for auto-correlation (relevant only for SubspaceNet model).
        model_type (str): The type of the model.
        save_datasets (bool, optional): Specifies whether to save the dataset. Defaults to False.
        datasets_path (Path, optional): The path for saving the dataset. Defaults to None.
        true_doa (list, optional): Predefined angles. Defaults to None.
        phase (str, optional): The phase of the dataset (test or training phase for CNN model). Defaults to None.

    Returns:
    --------
        tuple: A tuple containing the desired dataset comprised of (X-samples, Y-labels).

    """
    generic_dataset = []
    model_dataset = []
    samples_model = Samples(system_model_params)
    # Generate permutations for CNN model training dataset
    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        # NOTE: the full combinatorial set C(361, M) can be enormous
        # (e.g. ~7.78 million for M=3) and materializing every one of them
        # in RAM (as this used to do) can exhaust memory and crash the
        # process partway through. Instead, randomly sample `samples_size`
        # distinct M-angle combinations directly, without ever building the
        # full combinatorial list.
        angles_grid = np.linspace(start=-90, stop=90, num=361)
        max_combinations = math.comb(len(angles_grid), system_model_params.M)
        n_target = min(samples_size, max_combinations)
        doa_permutations_set = set()
        while len(doa_permutations_set) < n_target:
            combo = tuple(
                sorted(random.sample(list(angles_grid), system_model_params.M))
            )
            doa_permutations_set.add(combo)
        doa_permutations = [list(c) for c in doa_permutations_set]

    if model_type.startswith("DeepCNN") and phase.startswith("train"):
        for i, doa in tqdm(enumerate(doa_permutations), total=len(doa_permutations)):
            # Samples model creation
            samples_model.set_doa(doa)
            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            X_model = create_cov_tensor(X)
            # Ground-truth creation
            Y = torch.zeros_like(torch.tensor(angles_grid, dtype=torch.float32))
            for angle in doa:
                Y[list(angles_grid).index(angle)] = 1
            model_dataset.append((X_model, Y))
            generic_dataset.append((X, Y))
    else:
        for i in tqdm(range(samples_size)):
            # Samples model creation
            samples_model.set_doa(true_doa)
            # Observations matrix creation
            X = torch.tensor(
                samples_model.samples_creation(
                    noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
                )[0],
                dtype=torch.complex64,
            )
            if model_type.startswith("SubspaceNet"):
                # Generate auto-correlation tensor
                X_model = create_autocorrelation_tensor(X, tau).to(torch.float)
            elif model_type.startswith("DeepCNN") and phase.startswith("test"):
                # Generate 3d covariance parameters tensor
                X_model = create_cov_tensor(X)
            else:
                X_model = X
            # Ground-truth creation
            Y = torch.tensor(samples_model.doa, dtype=torch.float64)
            generic_dataset.append((X, Y))
            model_dataset.append((X_model, Y))

    if save_datasets:
        model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
            system_model_params, samples_size
        )
        samples_model_filename = f"samples_model" + set_dataset_filename(
            system_model_params, samples_size
        )

        torch.save(obj=model_dataset, f=datasets_path / phase / model_dataset_filename)
        torch.save(
            obj=generic_dataset, f=datasets_path / phase / generic_dataset_filename
        )
        if phase.startswith("test"):
            torch.save(
                obj=samples_model, f=datasets_path / phase / samples_model_filename
            )

    return model_dataset, generic_dataset, samples_model


def create_observations(
    system_model_params: SystemModelParams, samples_size: int, true_doa: list = None,
    sir_range: tuple = None, coherent_fraction: float = None,
):
    """Draws samples_size observations once, as a list of (X, DoA [rad]).
    Every model's input is derived from these same X (see model_features),
    so all models train and test on identical data.
    sir_range=(low, high) [dB] draws SIR ~ U[low, high] per sample and makes the last
    source stronger than the others by -SIR dB (same convention as E9).
    coherent_fraction=p makes each sample coherent with probability p, non-coherent otherwise (X12)."""
    samples_model = Samples(system_model_params)
    equal_powers, nature = samples_model.params.source_power_db, samples_model.params.signal_nature
    observations = []
    for _ in tqdm(range(samples_size)):
        if coherent_fraction is not None:
            samples_model.params.signal_nature = "coherent" if np.random.rand() < coherent_fraction else "non-coherent"
        if sir_range is not None:
            samples_model.params.source_power_db = [0] * (samples_model.params.M - 1) + [-np.random.uniform(*sir_range)]
        samples_model.set_doa(true_doa)
        # Random source order (set_doa sorts), so a per-source power offset
        # (source_power_db) lands on a random angle, not always the largest one
        samples_model.doa = np.random.permutation(samples_model.doa)
        X = torch.tensor(
            samples_model.samples_creation(
                noise_mean=0, noise_variance=1, signal_mean=0, signal_variance=1
            )[0],
            dtype=torch.complex64,
        )
        observations.append((X, torch.tensor(samples_model.doa, dtype=torch.float64)))
    # do not leave the last draw in the caller's params
    samples_model.params.source_power_db, samples_model.params.signal_nature = equal_powers, nature
    return observations, samples_model


def model_features(observations: list, model_type: str, tau: int = None, grid_labels=False):
    """Derives one model's (input, label) pairs from shared observations.
    grid_labels=True gives DeepCNN its training target: a multi-hot vector over the
    361-point 0.5-degree grid, with each DoA mapped to its nearest bin."""
    dataset = []
    for X, doa in observations:
        if model_type.startswith("SubspaceNet"):
            x = create_autocorrelation_tensor(X, tau).to(torch.float)
        elif model_type.startswith("DeepCNN"):
            x = create_cov_tensor(X)
        else:
            x = X
        y = doa
        if grid_labels:
            y = torch.zeros(361, dtype=torch.float32)
            y[torch.round((torch.rad2deg(doa) + 90) / 0.5).long()] = 1
        dataset.append((x, y))
    return dataset


# def read_data(Data_path: str) -> torch.Tensor:
def read_data(path: str):
    """
    Reads data from a file specified by the given path.

    Args:
    -----
        path (str): The path to the data file.

    Returns:
    --------
        torch.Tensor: The loaded data.

    Raises:
    -------
        None

    Examples:
    ---------
        >>> path = "data.pt"
        >>> read_data(path)

    """
    assert isinstance(path, (str, Path))
    # PyTorch >= 2.6 defaults torch.load's weights_only to True, which blocks
    # unpickling custom classes like src.signal_creation.Samples. These files
    # are ones you generated yourself with this codebase's own dataset
    # creation pipeline (not untrusted external files), so weights_only=False
    # is safe here.
    data = torch.load(path, weights_only=False)
    return data


# def autocorrelation_matrix(X: torch.Tensor, lag: int) -> torch.Tensor:
def autocorrelation_matrix(X: torch.Tensor, lag: int):
    """
    Computes the autocorrelation matrix for a given lag of the input samples.

    Args:
    -----
        X (torch.Tensor): Samples matrix input with shape [N, T].
        lag (int): The requested delay of the autocorrelation calculation.

    Returns:
    --------
        torch.Tensor: The autocorrelation matrix for the given lag.

    """
    Rx_lag = torch.zeros(X.shape[0], X.shape[0], dtype=torch.complex128)
    n_terms = X.shape[1] - lag
    if n_terms <= 0:
        # Structurally undefined: this lag exceeds the number of available
        # snapshots (T <= lag), e.g. tau > T scenarios (very few
        # observations). Return a zero matrix instead of dividing by zero
        # or a negative number (which produced NaN/Inf and crashed
        # downstream eigendecompositions). This degrades gracefully
        # (missing lag information -> less accurate estimate) rather than
        # crashing -- which is the expected, realistic behavior for a
        # low-snapshot regime.
        Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
        return Rx_lag
    # Vectorized form of sum_t (x_t - mean) (conj(x_{t+lag}) - mean)^T / n_terms,
    # identical to the former per-t loop (same scalar mean, same conjugation order)
    X = X.to(torch.complex128)
    mean = torch.mean(X)
    Rx_lag = (X[:, :n_terms] - mean) @ (torch.conj(X[:, lag : lag + n_terms]) - mean).T
    Rx_lag = Rx_lag / n_terms
    Rx_lag = torch.cat((torch.real(Rx_lag), torch.imag(Rx_lag)), 0)
    return Rx_lag


# def create_autocorrelation_tensor(X: torch.Tensor, tau: int) -> torch.Tensor:
def create_autocorrelation_tensor(X: torch.Tensor, tau: int):
    """
    Returns a tensor containing all the autocorrelation matrices for lags 0 to tau.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (BS, N, T).
        tau (int): Maximal time difference for the autocorrelation tensor.

    Returns:
    --------
        torch.Tensor: Tensor containing all the autocorrelation matrices,
                    with size (Batch size, tau, 2N, N).

    Raises:
    -------
        None

    """
    Rx_tau = []
    for i in range(tau):
        Rx_tau.append(autocorrelation_matrix(X, lag=i))
    Rx_autocorr = torch.stack(Rx_tau, dim=0)
    return Rx_autocorr


# def create_cov_tensor(X: torch.Tensor) -> torch.Tensor:
def create_cov_tensor(X: torch.Tensor):
    """
    Creates a 3D tensor of size (NxNx3) containing the real part, imaginary part, and phase component of the covariance matrix.

    Args:
    -----
        X (torch.Tensor): Observation matrix input with size (N, T).

    Returns:
    --------
        Rx_tensor (torch.Tensor): Tensor containing the auto-correlation matrices, with size (Batch size, N, N, 3).

    Raises:
    -------
        None

    """
    Rx = torch.cov(X)
    Rx_tensor = torch.stack((torch.real(Rx), torch.imag(Rx), torch.angle(Rx)), 2)
    return Rx_tensor


def load_datasets(
    system_model_params: SystemModelParams,
    model_type: str,
    samples_size: float,
    datasets_path: Path,
    train_test_ratio: float,
    is_training: bool = False,
):
    """
    Load different datasets based on the specified parameters and phase.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        model_type (str): The type of the model.
        samples_size (float): The size of the overall dataset.
        datasets_path (Path): The path to the datasets.
        train_test_ratio (float): The ration between train and test datasets.
        is_training (bool): Specifies whether to load the training dataset.

    Returns:
    --------
        List: A list containing the loaded datasets.

    """
    datasets = []
    # Define test set size
    test_samples_size = int(train_test_ratio * samples_size)
    # Generate datasets filenames
    model_dataset_filename = f"{model_type}_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    generic_dataset_filename = f"Generic_DataSet" + set_dataset_filename(
        system_model_params, test_samples_size
    )
    samples_model_filename = f"samples_model" + set_dataset_filename(
        system_model_params, test_samples_size
    )

    # Whether to load the training dataset
    if is_training:
        # Load training dataset
        try:
            model_trainingset_filename = f"{model_type}_DataSet" + set_dataset_filename(
                system_model_params, samples_size
            )
            train_dataset = read_data(
                datasets_path / "train" / model_trainingset_filename
            )
            datasets.append(train_dataset)
        except:
            raise Exception("load_datasets: Training dataset doesn't exist")
    # Load test dataset
    try:
        test_dataset = read_data(datasets_path / "test" / model_dataset_filename)
        datasets.append(test_dataset)
    except:
        raise Exception("load_datasets: Test dataset doesn't exist")
    # Load generic test dataset
    try:
        generic_test_dataset = read_data(
            datasets_path / "test" / generic_dataset_filename
        )
        datasets.append(generic_test_dataset)
    except:
        raise Exception("load_datasets: Generic test dataset doesn't exist")
    # Load samples models
    try:
        samples_model = read_data(datasets_path / "test" / samples_model_filename)
        datasets.append(samples_model)
    except:
        raise Exception("load_datasets: Samples model dataset doesn't exist")
    return datasets


def set_dataset_filename(system_model_params: SystemModelParams, samples_size: float):
    """Returns the generic suffix of the datasets filename.

    Args:
    -----
        system_model_params (SystemModelParams): an instance of SystemModelParams.
        samples_size (float): The size of the overall dataset.

    Returns:
    --------
        str: Suffix dataset filename
    """
    suffix_filename = (
        f"_{system_model_params.signal_type}_"
        + f"{system_model_params.signal_nature}_{samples_size}_M={system_model_params.M}_"
        + f"N={system_model_params.N}_T={system_model_params.T}_SNR={system_model_params.snr}_"
        + f"eta={system_model_params.eta}_sv_noise_var{system_model_params.sv_noise_var}_"
        + f"bias={system_model_params.bias}_"
        + ".h5"
    )
    return suffix_filename
