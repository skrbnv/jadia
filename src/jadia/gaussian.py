import numpy as np


def gaussian_kernel(size, sigma):
    """Generates a 1D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x: (1 / (np.sqrt(2 * np.pi) * sigma))
        * np.exp((-((x - (size - 1) / 2) ** 2)) / (2 * sigma**2)),
        (size,),
    )
    return kernel / np.sum(kernel)


def convolve1d(input_array, kernel):
    """Convolves a 1D input array with a 1D kernel."""
    input_size = len(input_array)
    kernel_size = len(kernel)
    pad_width = kernel_size // 2

    # Pad the input array with reflect mode
    padded_input = np.pad(input_array, pad_width, mode="reflect")

    # Initialize the output array
    output_array = np.zeros(input_size)

    # Perform convolution
    for i in range(input_size):
        output_array[i] = np.sum(padded_input[i : i + kernel_size] * kernel)

    return output_array


def gaussian_filter1d(input_array, sigma):
    """Applies a 1D Gaussian filter to a 1D input array."""
    # Define the size of the kernel (6 * sigma is a common choice)
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    # Generate Gaussian kernel
    kernel = gaussian_kernel(kernel_size, sigma)

    # Apply convolution
    return convolve1d(input_array, kernel)


if __name__ == "__main__":
    pass
