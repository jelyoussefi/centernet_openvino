# CenterNet Windows 11 Implementation
# A PowerShell script to replace the Linux/Docker Makefile

# ----------------------------------
# General Settings
# ----------------------------------
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Virtual Environment Settings
$VENV_NAME = "centernet_venv"
$VENV_DIR = Join-Path $SCRIPT_DIR $VENV_NAME
$PYTHON_VERSION = "3.11"  # Python 11 (3.11)
$REQUIREMENTS_FILE = Join-Path $SCRIPT_DIR "requirements.txt"

# Directories
$DATASET_DIR = Join-Path $SCRIPT_DIR "dataset"
$MODELS_DIR = Join-Path $SCRIPT_DIR "models"
$COCO_DATASET_DIR = Join-Path $SCRIPT_DIR "datasets\COCO"

# Default Parameters
$DEVICE = if ($env:DEVICE) { $env:DEVICE } else { "GPU" }
$INPUT_SIZE = if ($env:INPUT_SIZE) { $env:INPUT_SIZE } else { 640 }
$PRECISION = if ($env:PRECISION) { $env:PRECISION } else { "FP32" }
$CHECKPOINT_PATH = if ($env:CHECKPOINT_PATH) { $env:CHECKPOINT_PATH } else { Join-Path $SCRIPT_DIR "model_epoch_8.pth" }
$CONFIG_PATH = if ($env:CONFIG_PATH) { $env:CONFIG_PATH } else { Join-Path $SCRIPT_DIR "model_centernet_r18_8xb16-crop512-140e_coco.py" }

# Model and Input Paths
$MODEL_PATH = Join-Path $MODELS_DIR "$PRECISION\centernet.xml"
$INPUT_PATH = Join-Path $SCRIPT_DIR "streams\tube_capped.jpg"

function Setup-VirtualEnvironment {
    Write-Host "🔧 Setting up Python $PYTHON_VERSION virtual environment..."
    
    # Check if Python is installed
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        Write-Host "Error: Python is not installed or not in the PATH" -ForegroundColor Red
        Write-Host "Please install Python $PYTHON_VERSION from https://www.python.org/downloads/" -ForegroundColor Yellow
        return $false
    }
    
    # Check Python version
    $pythonVersion = python --version 2>&1
    if (-not $pythonVersion.ToString().Contains("Python 3")) {
        Write-Host "Error: Python 3 is required" -ForegroundColor Red
        return $false
    }
    
    # Check if virtual environment exists
    if (Test-Path $VENV_DIR) {
        Write-Host "Virtual environment already exists at $VENV_DIR" -ForegroundColor Yellow
        $recreate = Read-Host "Do you want to recreate it? (y/n)"
        if ($recreate -eq "y") {
            Remove-Item -Path $VENV_DIR -Recurse -Force
        } else {
            Write-Host "Using existing virtual environment" -ForegroundColor Green
            return $true
        }
    }
    
    # Install virtualenv if needed
    Write-Host "Installing/Upgrading virtualenv..."
    python -m pip install --upgrade virtualenv
    
    # Create virtual environment
    Write-Host "Creating virtual environment at $VENV_DIR..."
    python -m virtualenv $VENV_DIR --python=$PYTHON_VERSION
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Virtual environment created successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Failed to create virtual environment" -ForegroundColor Red
        Write-Host "You might need to install Python $PYTHON_VERSION first" -ForegroundColor Yellow
        return $false
    }
}

function Enter-VirtualEnvironment {
    # Check if virtual environment exists
    if (-not (Test-Path $VENV_DIR)) {
        Write-Host "Virtual environment not found at $VENV_DIR" -ForegroundColor Red
        $create = Read-Host "Do you want to create it now? (y/n)"
        if ($create -eq "y") {
            if (-not (Setup-VirtualEnvironment)) {
                return $false
            }
        } else {
            return $false
        }
    }
    
    # Activate virtual environment
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        Write-Host "Virtual environment activated" -ForegroundColor Green
        return $true
    } else {
        Write-Host "Activation script not found at $activateScript" -ForegroundColor Red
        return $false
    }
}
function Show-Menu {
    Clear-Host
    Write-Host "================ CenterNet OpenVINO on Windows 11 ================"
    Write-Host "1: Setup Python 3.11 Virtual Environment"
    Write-Host "2: Install Dependencies (pip install -r requirements.txt)"
    Write-Host "3: Run Inference"
    Write-Host "4: Prepare COCO Dataset"
    Write-Host "5: Export PyTorch Model to OpenVINO"
    Write-Host "6: Quantize Model to INT8"
    Write-Host "Q: Quit"
    Write-Host "=============================================================="
}

function Run-Inference {
    Write-Host "Running CenterNet Inference demo in $PRECISION ..."
    
    # Make sure we're in the virtual environment
    if (-not (Enter-VirtualEnvironment)) {
        Write-Host "Cannot run inference without virtual environment" -ForegroundColor Yellow
        return
    }
    
    # Run the detector script
    python "$SCRIPT_DIR\detector.py" `
        -m $MODEL_PATH `
        -i $INPUT_PATH `
        -d $DEVICE `
        -p $PRECISION
}

function Prepare-Dataset {
    Write-Host "🚀 Preparing the COCO dataset ..."
    
    # Make sure we're in the virtual environment
    if (-not (Enter-VirtualEnvironment)) {
        Write-Host "Cannot prepare dataset without virtual environment" -ForegroundColor Yellow
        return
    }
    
    # Create directory if it doesn't exist
    if (-not (Test-Path $COCO_DATASET_DIR)) {
        New-Item -ItemType Directory -Path $COCO_DATASET_DIR -Force | Out-Null
    }
    
    # Run the dataset preparation script
    python "$SCRIPT_DIR\prepare_coco_dataset.py"
}

function Export-ToOpenVINO {
    Write-Host "Exporting PyTorch model to OpenVINO ..."
    
    # Make sure we're in the virtual environment
    if (-not (Enter-VirtualEnvironment)) {
        Write-Host "Cannot export model without virtual environment" -ForegroundColor Yellow
        return
    }
    
    # Create output directory if it doesn't exist
    $outputDir = Join-Path $MODELS_DIR "FP32"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Run the export script
    python "$SCRIPT_DIR\openvino_export.py" `
        --config $CONFIG_PATH `
        --checkpoint $CHECKPOINT_PATH `
        --resolution $INPUT_SIZE `
        --output_dir $outputDir
}

function Quantize-Model {
    Write-Host "⚙️ Quantizing model to INT8 ..."
    
    # Make sure we're in the virtual environment
    if (-not (Enter-VirtualEnvironment)) {
        Write-Host "Cannot quantize model without virtual environment" -ForegroundColor Yellow
        return
    }
    
    # Make sure the dataset is prepared
    if (-not (Test-Path $COCO_DATASET_DIR)) {
        Write-Host "⚠️ COCO dataset not found. Preparing dataset first..." -ForegroundColor Yellow
        Prepare-Dataset
    }
    
    # Create output directory if it doesn't exist
    $outputDir = Join-Path $MODELS_DIR "INT8"
    if (-not (Test-Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    }
    
    # Run the quantization script
    python "$SCRIPT_DIR\quantize.py" `
        --model (Join-Path $MODELS_DIR "FP32\centernet.xml") `
        --dataset $COCO_DATASET_DIR `
        --resize $INPUT_SIZE $INPUT_SIZE `
        --output (Join-Path $outputDir "centernet.xml")
}

# ----------------------------------
# Main Script
# ----------------------------------

# Check for command line arguments
if ($args.Count -gt 0) {
    switch ($args[0].ToLower()) {
        "setup" { Setup-VirtualEnvironment; exit }
        "install" { Install-Dependencies; exit }
        "run" { Run-Inference; exit }
        "dataset" { Prepare-Dataset; exit }
        "export" { Export-ToOpenVINO; exit }
        "quantize" { Quantize-Model; exit }
        default { 
            Write-Host "Unknown command: $($args[0])"
            Write-Host "Available commands: setup, install, run, dataset, export, quantize"
            exit 1
        }
    }
}

# If no arguments, show interactive menu
do {
    Show-Menu
    $selection = Read-Host "Please make a selection"
    
    switch ($selection) {
        '1' { Setup-VirtualEnvironment; pause }
        '2' { Install-Dependencies; pause }
        '3' { Run-Inference; pause }
        '4' { Prepare-Dataset; pause }
        '5' { Export-ToOpenVINO; pause }
        '6' { Quantize-Model; pause }
        'q' { return }
    }
} until ($selection -eq 'q')
