import torch
import torch.nn.functional as F

def create_gaussian_kernel(kernel_size=9, sigma=3, channels=3):
    """
    Create a Gaussian kernel.

    Args:
        kernel_size (int): Size of the Gaussian kernel (odd integer).
        sigma (float): Standard deviation of the Gaussian distribution.
        channels (int): Number of channels in the image.

    Returns:
        torch.Tensor: Gaussian kernel tensor of shape (channels, 1, kernel_size, kernel_size).
    """
    # Create a 1D Gaussian kernel
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2
    variance = sigma ** 2

    # Calculate the 2D Gaussian kernel
    gaussian_kernel = (1.0 / (2 * torch.pi * variance)) * \
        torch.exp(
            -((xy_grid - mean) ** 2).sum(dim=-1) / (2 * variance)
        )

    # Normalize the kernel
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    # Reshape to depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel.to(torch.float16)

def create_motion_blur_kernel(kernel_size=15, angle=0):
    """
    Create a motion blur kernel.

    Args:
        kernel_size (int): Size of the kernel (odd integer).
        angle (float): Angle in degrees to rotate the kernel.

    Returns:
        torch.Tensor: Motion blur kernel of shape (1, 1, kernel_size, kernel_size).
    """
    # Create a linear kernel (horizontal line)
    kernel = torch.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = torch.ones(kernel_size)
    kernel = kernel / kernel_size  # Normalize

    # Rotate the kernel
    theta = torch.tensor([
        [torch.cos(torch.deg2rad(torch.tensor(angle))), -torch.sin(torch.deg2rad(torch.tensor(angle))), 0],
        [torch.sin(torch.deg2rad(torch.tensor(angle))),  torch.cos(torch.deg2rad(torch.tensor(angle))), 0]
    ])
    grid = F.affine_grid(theta.unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).size())
    kernel = F.grid_sample(kernel.unsqueeze(0).unsqueeze(0), grid)
    return kernel.to(torch.float16)


def apply_motion_blur_pytorch(images, kernel_size=15, angle=0):
    """
    Apply motion blur to a batch of images.

    Args:
        images (torch.Tensor): Images of shape (batch, channels, height, width).
        kernel_size (int): Size of the motion blur kernel.
        angle (float): Angle for motion blur in degrees.

    Returns:
        torch.Tensor: Motion blurred images.
    """
    batch_size, channels, height, width = images.shape

    # Create motion blur kernel
    kernel = create_motion_blur_kernel(kernel_size, angle)
    kernel = kernel.to(images.device)

    # Convolve each channel with the kernel
    padding = kernel_size // 2
    # breakpoint()
    blurred_images = torch.cat([F.conv2d(images[:, i:i+1], kernel, padding=padding) for i in range(images.shape[1])], dim=1)
    blurred_images = blurred_images.view(batch_size, channels, height, width)
    return blurred_images


def apply_gaussian_blur_pytorch(images, kernel_size=9, sigma=3):
    """
    Apply Gaussian blur to a batch of images using convolution.

    Args:
        images (torch.Tensor): Images of shape (batch, channels, height, width).
        kernel_size (int): Size of the Gaussian kernel (odd integer).
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: Blurred images.
    """
    batch_size, channels, height, width = images.shape
    device = images.device

    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, channels).to(device)

    # Apply Gaussian blur using depthwise convolution
    padding = kernel_size // 2
    blurred_images = F.conv2d(images, kernel, padding=padding, groups=channels)
    return blurred_images


def add_gaussian_noise_pytorch(images, mean=0.0, std=0.01):
    """
    Add Gaussian noise to images.

    Args:
        images (torch.Tensor): Images of shape (batch, channels, height, width).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Noisy images.
    """
    noise = torch.randn_like(images) * std + mean
    noisy_images = images + noise
    # noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images


def add_poisson_noise_pytorch(images, lam=0.05):
    """
    Add Poisson noise to images.

    Args:
        images (torch.Tensor): Images of shape (batch, channels, height, width) with values in [0, 1].
        lam (float): Lambda parameter for Poisson distribution.

    Returns:
        torch.Tensor: Noisy images.
    """
    # Scale images to a higher range for Poisson distribution
    images_scaled = images * 255.0
    noisy_images = torch.poisson(images_scaled)
    noisy_images = noisy_images / 255.0
    # noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images


def degrade_image(
    images,
    scale_factor=2,
    apply_blur=False,
    blur_kernel_size=9,
    blur_sigma=3,
    apply_motion_blur=False,
    motion_blur_kernel_size=15,
    motion_blur_angle=15,
    noise_type="gaussian",
    noise_std=1.0
):
    """
    Degrade images by applying multiple degradation operations.

    Args:
        images (torch.Tensor): HR images of shape (batch, channels, height, width) with values in [0, 1].
        scale_factor (int): Downsampling scale factor.
        apply_blur (bool): Whether to apply Gaussian blur.
        blur_kernel_size (int): Gaussian kernel size.
        blur_sigma (float): Gaussian kernel standard deviation.
        apply_motion_blur (bool): Whether to apply motion blur.
        motion_blur_kernel_size (int): Motion blur kernel size.
        motion_blur_angle (float): Angle for motion blur.
        noise_type (str): 'gaussian', 'poisson', or None.
        noise_std (float): Standard deviation for Gaussian noise.

    Returns:
        torch.Tensor: Degraded LR images.
    """
    degraded = images.clone()
    assert images.shape[2] % scale_factor == 0 and images.shape[3] % scale_factor == 0

    # Apply Gaussian Blur
    if apply_blur:
        degraded = apply_gaussian_blur_pytorch(degraded, kernel_size=blur_kernel_size, sigma=blur_sigma)

    # Apply Motion Blur
    if apply_motion_blur:
        degraded = apply_motion_blur_pytorch(degraded, kernel_size=motion_blur_kernel_size, angle=motion_blur_angle)

    # Downsample
    degraded = F.interpolate(degraded, scale_factor=1/scale_factor, mode='bicubic', align_corners=False)
    degraded = F.interpolate(degraded, scale_factor=scale_factor, mode='bicubic', align_corners=False)

    # Add Noise
    if noise_type == 'gaussian':
        degraded = add_gaussian_noise_pytorch(degraded, mean=0.0, std=noise_std)
    elif noise_type == 'poisson':
        degraded = add_poisson_noise_pytorch(degraded)

    return degraded
