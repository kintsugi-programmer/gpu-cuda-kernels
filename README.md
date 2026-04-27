# GPU CUDA Edge Detection

A high-performance edge detection application that leverages NVIDIA CUDA and cuDNN to accelerate image and video processing. This project demonstrates GPU-accelerated convolution operations, CUDA graphs, and optimized memory management for real-time edge detection.

## Features

- **GPU-Accelerated Processing**: Utilizes NVIDIA CUDA and cuDNN for fast convolution operations
- **CUDA Graphs**: Implements CUDA graph execution for optimized kernel launch overhead
- **Multi-Format Support**: Process both static images (PNG) and videos (MP4)
- **Real-Time Timing**: Built-in profiling for total, processing, and GPU-only execution times
- **Flexible CLI**: Command-line interface for specifying input/output files
- **Optimized Memory Management**: Custom GPU blob abstraction for efficient device memory handling

## File Structure

```
gpu-cuda-kernels/
├── Makefile                    # Build configuration
├── README.md                   # This file
├── LICENSE                     # Project license (MIT)
├── data/                       # Test data directory
│   ├── Lena.png               # Sample input image
│   └── Lena_edge.png          # Reference output image
├── include/                    # Header files
│   ├── cli.hpp                # Command-line interface parsing
│   ├── convolution.hpp        # cuDNN convolution wrapper template
│   ├── cuda_graph.hpp         # CUDA graph execution support
│   ├── cuda_kernels.hpp       # CUDA kernel declarations
│   ├── filter.hpp             # Edge detection filter implementation
│   ├── gpu_blob.hpp           # GPU memory blob abstraction
│   ├── gpu_session.hpp        # CUDA/cuDNN session management
│   ├── helper_cuda.h          # NVIDIA CUDA helper utilities
│   ├── helper_string.h        # NVIDIA string helper utilities
│   ├── image_manip.hpp        # Image manipulation kernel declarations
│   ├── io.hpp                 # Image I/O operations header
│   ├── timer.hpp              # CUDA event-based timer
│   └── types.hpp              # Core type definitions (ImageCPU, ImageGPU, Kernel)
└── src/                        # Source files
    ├── cuda_kernels.cu         # CUDA convolution and edge detection kernels
    ├── edgeDetection.cpp       # Main application entry point
    ├── gpu_blob.cu             # GPU memory blob implementation
    ├── image_manip.cu          # Image manipulation CUDA kernels
    └── io.cpp                  # Image loading/saving with OpenCV
```

## Dependencies

- **CUDA Toolkit** (with nvcc compiler)
- **cuDNN** (NVIDIA CUDA Deep Neural Network library)
- **OpenCV 4** (with FreeImage support)
- **C++17** compatible compiler
- **pkg-config** (for OpenCV detection)

### Installation on macOS (Homebrew)

```bash
brew install cuda cudnn opencv pkg-config
```

### Installation on Ubuntu/Debian

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit libcudnn8 libcudnn8-dev libopencv-dev pkg-config
```

## Build Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd gpu-cuda-kernels
   ```

2. Update the `CUDA_PATH` in the Makefile if your CUDA installation is in a non-standard location (default: `/usr/local/cuda`).

3. Build the project:
   ```bash
   make
   ```

   This will create the `bin/` directory with the `edgeDetection` executable.

4. (Optional) Generate compile_commands.json for clangd/IDE support:
   ```bash
   make compile_commands
   ```

## Usage

### Basic Usage

Process the default sample image (data/Lena.png):
```bash
make run
```

### Command-Line Options

```bash
./bin/edgeDetection --input <input_file> --output <output_file>
```

**Supported input formats:**
- PNG images (`.png`)
- MP4 videos (`.mp4`)

**Examples:**

Process a PNG image:
```bash
./bin/edgeDetection --input data/Lena.png --output data/Lena_edge.png
```

Process a video:
```bash
./bin/edgeDetection --input input_video.mp4 --output output_video.mp4
```

If no input is specified, the application defaults to `data/Lena.png`.

## Example Results

| Input Image | Output Image |
|-------------|--------------|
| ![Lena Original](data/Lena.png) | ![Lena Edge Detection](data/Lena_edge.png) |

## How It Works

The edge detection pipeline follows these steps:

1. **Load Input**: Image/video frames are loaded from disk using OpenCV
2. **Convert to Float**: Input images (UINT8, 4 channels) are converted to float format
3. **Grayscale Conversion**: RGB channels are convolved with weights `[0.299, 0.587, 0.114]` to produce grayscale
4. **Edge Detection**: A 3x3 Sobel-like kernel detects edges in both X and Y directions
5. **Non-Maximum Suppression**: Pointwise absolute value and 2D-to-1D reduction
6. **Smoothing**: Multiple iterations of 3x3 Gaussian smoothing
7. **Edge Enhancement**: 5x5 kernel enhances remaining edges
8. **Reconstruction**: Edges are broadcast back to 4 channels and merged with original image
9. **Output**: Result is converted back to UINT8 and saved

### Key Components

| Component | Description |
|-----------|-------------|
| `GpuSession` | Manages CUDA and cuDNN context/handle lifecycle |
| `GpuBlob` | RAII wrapper for CUDA device memory allocations |
| `ImageCPU<T, C>` | CPU-side image container (T: type, C: channels) |
| `ImageGPU<T, C>` | GPU-side image container with automatic memory management |
| `Kernel<T, F, W, H, C>` | Compile-time defined convolution kernel |
| `Convolution<K, I, O>` | Template class wrapping cuDNN convolution operations |
| `Filter` | High-level edge detection filter orchestrating the pipeline |
| `CudaGraph` | Wraps CUDA graph for reduced kernel launch overhead |
| `Timer` | CUDA event-based timer for performance measurement |

## Performance

The application provides timing breakdown:
- **Total time** (incl. I/O): End-to-end processing time
- **Processing time** (excl. I/O): GPU processing only
- **GPU time**: Pure CUDA kernel execution time

Example output:
```
Elapsed time in nanoseconds:
        		  Total		   per frame
incl. io		 5000000000	   50000000
excl. io		 4500000000	   45000000
gpu				 4000000000	   40000000
```

## Code Quality

Run clang-tidy for static analysis:
```bash
make tidy
```

Clean build artifacts:
```bash
make clean
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved. (portions of helper_cuda.h and helper_string.h)

## Acknowledgments

- NVIDIA for the CUDA toolkit and helper utilities
- OpenCV for image and video processing capabilities
- cuDNN for optimized deep learning convolution operations
