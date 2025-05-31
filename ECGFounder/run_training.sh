#!/bin/bash

# ECGFounder Fine-tuning Script
# This script runs fine-tuning on 6 ECG conditions: NORM, AFIB, AFLT, 1dAVb, RBBB, LBBB

echo "🚀 Starting ECGFounder Fine-tuning on 6 Conditions"
echo "=================================================="

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "ECGFounder" ]]; then
    echo "⚠️  Warning: ECGFounder conda environment not activated"
    echo "Please run: conda activate ECGFounder"
    exit 1
fi

# Check if checkpoint files exist
if [ ! -f "./checkpoint/12_lead_ECGFounder.pth" ]; then
    echo "❌ 12-lead checkpoint not found!"
    echo "Please download from: https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main"
    echo "Or run: wget https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth -P checkpoint/"
    exit 1
fi

echo "✅ Checkpoint files found"

# Configuration options
MODEL_TYPE="12lead"  # or "1lead"
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.001
MAX_SAMPLES=1000  # Set to higher number or remove flag for all data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --linear_prob)
            LINEAR_PROB="--linear_prob"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model_type     Model type: 12lead or 1lead (default: 12lead)"
            echo "  --batch_size     Batch size (default: 32)"
            echo "  --epochs         Number of epochs (default: 50)"
            echo "  --lr             Learning rate (default: 0.001)"
            echo "  --max_samples    Max samples per condition (default: 1000)"
            echo "  --linear_prob    Use linear probing (freeze all except last layer)"
            echo "  --help           Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model_type 12lead --epochs 30 --batch_size 16 --linear_prob"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "   Model Type: $MODEL_TYPE"
echo "   Batch Size: $BATCH_SIZE"
echo "   Epochs: $EPOCHS"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Max Samples per Condition: $MAX_SAMPLES"
echo "   Linear Probing: ${LINEAR_PROB:-No}"
echo ""

# Run the training
echo "🏃 Starting training..."
python train_6_conditions.py \
    --model_type $MODEL_TYPE \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_samples $MAX_SAMPLES \
    $LINEAR_PROB

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📊 Check the results directory for:"
    echo "   - Training plots"
    echo "   - Confusion matrices"
    echo "   - Classification reports"
    echo "   - Best model checkpoint"
else
    echo ""
    echo "❌ Training failed!"
    echo "Please check the error messages above"
    exit 1
fi 