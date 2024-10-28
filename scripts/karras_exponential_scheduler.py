from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential
from modules import sd_schedulers, errors
from functools import partial
from pathlib import Path
import random
import torch
import yaml


extension_root = Path(__file__).parents[1]
kes_config_dir = extension_root / 'kes-configs'


def get_random_or_default(scheduler_config, key_prefix, default_value, global_randomize):
    """Helper function to either randomize a value based on conditions or return the default."""
    randomize_flag = global_randomize or scheduler_config.get(f'{key_prefix}_rand', False)
    if randomize_flag:
        rand_min = scheduler_config.get(f'{key_prefix}_rand_min', default_value * 0.8)
        rand_max = scheduler_config.get(f'{key_prefix}_rand_max', default_value * 1.2)
        value = random.uniform(rand_min, rand_max)
    else:
        value = default_value

    return value


default_config = {
    "debug": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "sigma_min": 0.01,
    "sigma_max": 50,  # if sigma_max is too low the resulting picture may be undesirable.
    "start_blend": 0.1,
    "end_blend": 0.5,
    "sharpness": 0.95,
    "early_stopping_threshold": 0.01,
    "update_interval": 10,
    "initial_step_size": 0.9,
    "final_step_size": 0.2,
    "initial_noise_scale": 1.25,
    "final_noise_scale": 0.8,
    "smooth_blend_factor": 11,
    "step_size_factor": 0.8,  # suggested value to avoid over smoothing
    "noise_scale_factor": 0.9,  # suggested value to add more variation
    "randomize": False,
    "sigma_min_rand": False,
    "sigma_min_rand_min": 0.001,
    "sigma_min_rand_max": 0.05,
    "sigma_max_rand": False,
    "sigma_max_rand_min": 0.05,
    "sigma_max_rand_max": 0.20,
    "start_blend_rand": False,
    "start_blend_rand_min": 0.05,
    "start_blend_rand_max": 0.2,
    "end_blend_rand": False,
    "end_blend_rand_min": 0.4,
    "end_blend_rand_max": 0.6,
    "sharpness_rand": False,
    "sharpness_rand_min": 0.85,
    "sharpness_rand_max": 1.0,
    "early_stopping_rand": False,
    "early_stopping_rand_min": 0.001,
    "early_stopping_rand_max": 0.02,
    "update_interval_rand": False,
    "update_interval_rand_min": 5,
    "update_interval_rand_max": 10,
    "initial_step_rand": False,
    "initial_step_rand_min": 0.7,
    "initial_step_rand_max": 1.0,
    "final_step_rand": False,
    "final_step_rand_min": 0.1,
    "final_step_rand_max": 0.3,
    "initial_noise_rand": False,
    "initial_noise_rand_min": 1.0,
    "initial_noise_rand_max": 1.5,
    "final_noise_rand": False,
    "final_noise_rand_min": 0.6,
    "final_noise_rand_max": 1.0,
    "smooth_blend_factor_rand": False,
    "smooth_blend_factor_rand_min": 6,
    "smooth_blend_factor_rand_max": 11,
    "step_size_factor_rand": False,
    "step_size_factor_rand_min": 0.65,
    "step_size_factor_rand_max": 0.85,
    "noise_scale_factor_rand": False,
    "noise_scale_factor_rand_min": 0.75,
    "noise_scale_factor_rand_max": 0.95,
}


def simple_karras_exponential_scheduler(
        n, device, sigma_min=0.01, sigma_max=50, start_blend=0.1, end_blend=0.5,
        sharpness=0.95, early_stopping_threshold=0.01, update_interval=10, initial_step_size=0.9,
        final_step_size=0.2, initial_noise_scale=1.25, final_noise_scale=0.8, smooth_blend_factor=11, step_size_factor=0.8, noise_scale_factor=0.9, randomize=False,
        config_path=None,
):
    """
    Scheduler function that blends sigma sequences using Karras and Exponential methods with adaptive parameters.

    Parameters:
        n (int): Number of steps.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.
        device (torch.device): The device on which to perform computations (e.g., 'cuda' or 'cpu').
        start_blend (float): Initial blend factor for dynamic blending.
        end_bend (float): Final blend factor for dynamic blending.
        sharpen_factor (float): Sharpening factor to be applied adaptively.
        early_stopping_threshold (float): Threshold to trigger early stopping.
        update_interval (int): Interval to update blend factors.
        initial_step_size (float): Initial step size for adaptive step size calculation.
        final_step_size (float): Final step size for adaptive step size calculation.
        initial_noise_scale (float): Initial noise scale factor.
        final_noise_scale (float): Final noise scale factor.
        step_size_factor: Adjust to compensate for oversmoothing
        noise_scale_factor: Adjust to provide more variation

    Returns:
        torch.Tensor: A tensor of blended sigma values.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)['scheduler']
    except Exception:
        errors.report(f"Failed to load config file: {config_path}", exc_info=True)
        config = {}

    config = {**default_config, **config}
    diff = {key: config[key] for key in config if key not in default_config or config[key] != default_config[key]}
    from pprint import pprint
    print('config defaults diff:')
    pprint(diff)

    global_randomize = default_config.get('randomize', False)

    sigma_min = get_random_or_default(config, 'sigma_min', sigma_min, global_randomize)
    sigma_max = get_random_or_default(config, 'sigma_max', sigma_max, global_randomize)
    start_blend = get_random_or_default(config, 'start_blend', start_blend, global_randomize)
    end_blend = get_random_or_default(config, 'end_blend', end_blend, global_randomize)
    sharpness = get_random_or_default(config, 'sharpness', sharpness, global_randomize)
    early_stopping_threshold = get_random_or_default(config, 'early_stopping', early_stopping_threshold, global_randomize)
    update_interval = get_random_or_default(config, 'update_interval', update_interval, global_randomize)
    initial_step_size = get_random_or_default(config, 'initial_step', initial_step_size, global_randomize)
    final_step_size = get_random_or_default(config, 'final_step', final_step_size, global_randomize)
    initial_noise_scale = get_random_or_default(config, 'initial_noise', initial_noise_scale, global_randomize)
    final_noise_scale = get_random_or_default(config, 'final_noise', final_noise_scale, global_randomize)
    smooth_blend_factor = get_random_or_default(config, 'smooth_blend_factor', smooth_blend_factor, global_randomize)
    step_size_factor = get_random_or_default(config, 'step_size_factor', step_size_factor, global_randomize)
    noise_scale_factor = get_random_or_default(config, 'noise_scale_factor', noise_scale_factor, global_randomize)

    # Expand sigma_max slightly to account for smoother transitions
    sigma_max = sigma_max * 1.1
    # Generate sigma sequences using Karras and Exponential methods
    sigmas_karras = get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    sigmas_exponential = get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    # Match lengths of sigma sequences
    target_length = min(len(sigmas_karras), len(sigmas_exponential))
    sigmas_karras = sigmas_karras[:target_length]
    sigmas_exponential = sigmas_exponential[:target_length]

    if sigmas_karras is None:
        raise ValueError("Sigmas Karras:{sigmas_karras} Failed to generate or assign sigmas correctly.")
    if sigmas_exponential is None:
        raise ValueError("Sigmas Exponential: {sigmas_exponential} Failed to generate or assign sigmas correctly.")
        # sigmas_karras = torch.zeros(n).to(device)
        # sigmas_exponential = torch.zeros(n).to(device)

    # Define progress and initialize blend factor
    progress = torch.linspace(0, 1, len(sigmas_karras)).to(device)
    sigs = torch.zeros_like(sigmas_karras).to(device)
    # Iterate through each step, dynamically adjust blend factor, step size, and noise scaling
    for i in range(len(sigmas_karras)):
        # Adaptive step size and blend factor calculations
        step_size = initial_step_size * (1 - progress[i]) + final_step_size * progress[i] * step_size_factor  # 0.8 default value Adjusted to avoid over-smoothing
        dynamic_blend_factor = start_blend * (1 - progress[i]) + end_blend * progress[i]
        noise_scale = initial_noise_scale * (1 - progress[i]) + final_noise_scale * progress[i] * noise_scale_factor  # 0.9 default value Adjusted to keep more variation

        # Calculate smooth blending between the two sigma sequences
        smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * smooth_blend_factor)  # Increase scaling factor to smooth transitions more

        # Compute blended sigma values
        blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend

        # Apply step size and noise scaling
        sigs[i] = blended_sigma * step_size * noise_scale

    # Optional: Adaptive sharpening based on sigma values
    sharpen_mask = torch.where(sigs < sigma_min * 1.5, sharpness, 1.0).to(device)
    sigs = sigs * sharpen_mask

    # Implement early stop criteria based on sigma convergence
    change = torch.abs(sigs[1:] - sigs[:-1])
    if torch.all(change < early_stopping_threshold):
        return sigs[:len(change) + 1].to(device)

    if torch.isnan(sigs).any() or torch.isinf(sigs).any():
        raise ValueError("Invalid sigma values detected (NaN or Inf).")

    print(sigs)
    return sigs.to(device)


# add scheduler configs in kes_config_dir as usable schedulers into webui without modifying webui code
for kes_config in kes_config_dir.glob('*.yaml'):
    if kes_config.stem in sd_schedulers.schedulers_map:
        continue
    kes_scheduler = sd_schedulers.Scheduler(kes_config.stem, kes_config.stem, partial(simple_karras_exponential_scheduler, config_path=kes_config))
    sd_schedulers.schedulers.append(kes_scheduler)
    sd_schedulers.schedulers_map[kes_config.stem] = kes_scheduler
