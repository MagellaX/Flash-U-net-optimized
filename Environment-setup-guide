# Environment Setup for macOS and Linux

## Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git

## macOS Setup

```bash
# Create a new directory for your project
mkdir flash-unet-optimized
cd flash-unet-optimized

# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn einops datasets tqdm matplotlib numpy

# Clone the repository (if using git)
git clone https://github.com/yourusername/flash-unet-optimized.git

# Deactivate the environment when done
deactivate
```

## Linux Setup

```bash
# Create a new directory for your project
mkdir flash-unet-optimized
cd flash-unet-optimized

# Create a new virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn einops datasets tqdm matplotlib numpy

# Clone the repository (if using git)
git clone https://github.com/yourusername/flash-unet-optimized.git

# Deactivate the environment when done
deactivate
```

## Quick Commands Reference

### Activate Environment
macOS/Linux:
```bash
source venv/bin/activate
```

### Deactivate Environment
All platforms:
```bash
deactivate
```

### Check Installed Packages
```bash
pip list
```

### Export Dependencies
```bash
pip freeze > requirements.txt
```

### Install from requirements.txt
```bash
pip install -r requirements.txt
```

## Notes
- Make sure CUDA is properly installed if using GPU
- System might need additional dependencies based on your specific setup
- Consider adding your virtual environment directory to .gitignore:
```bash
echo "venv/" >> .gitignore
```

## Troubleshooting
If you encounter an error installing flash-attn:
```bash
# Install build dependencies first
pip install ninja
pip install flash-attn --no-build-isolation
```
