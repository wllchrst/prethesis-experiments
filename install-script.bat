@echo off

echo Creating new environment...
call conda create -n prethesis_environment python=3.12.8 -y

echo Activating environment...
call conda activate prethesis_environment

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo Installing PyTorch with CUDA 12.6...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo Setup complete.
