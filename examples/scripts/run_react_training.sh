#!/bin/bash

# React Training Workflow Automation Script
# This script automates the 3-step process for React model training with Q-Store

set -e  # Exit on error

echo "======================================================================"
echo "TinyLlama React Training with Q-Store Quantum Database"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to examples directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Step 1: Generate Dataset
echo -e "${BLUE}Step 1/3: Generating React training dataset...${NC}"
echo "Running: python -m q_store_examples.react_dataset_generator"
echo ""

python -m q_store_examples.react_dataset_generator

echo ""
echo -e "${GREEN}✓ Dataset generation complete!${NC}"
echo ""

# Step 2: Verify Dataset
echo -e "${BLUE}Step 2/3: Verifying dataset...${NC}"
echo ""

if [ -f "react_train.jsonl" ]; then
    SAMPLE_COUNT=$(cat react_train.jsonl | wc -l)
    echo "📊 Dataset Statistics:"
    echo "   Total samples: $SAMPLE_COUNT"
    echo ""
    
    # Count by type
    GEN_COUNT=$(grep -c '"instruction".*[Cc]reate\|[Bb]uild' react_train.jsonl || true)
    FIX_COUNT=$(grep -c '"instruction".*[Ff]ix' react_train.jsonl || true)
    EXP_COUNT=$(grep -c '"instruction".*[Ee]xplain' react_train.jsonl || true)
    CONV_COUNT=$(grep -c '"instruction".*[Cc]onvert' react_train.jsonl || true)
    
    echo "   Component Generation: ~$GEN_COUNT samples"
    echo "   Bug Fixing: ~$FIX_COUNT samples"
    echo "   Explanations: ~$EXP_COUNT samples"
    echo "   Conversions: ~$CONV_COUNT samples"
    echo ""
    
    # Show sample
    echo "📝 Sample entry:"
    head -n 1 react_train.jsonl | python -m json.tool | head -n 15
    echo "   ..."
    echo ""
    
    echo -e "${GREEN}✓ Dataset verification complete!${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: react_train.jsonl not found${NC}"
    exit 1
fi

echo ""

# Step 3: Run Training
echo -e "${BLUE}Step 3/3: Starting quantum-enhanced training...${NC}"
echo "Running: python -m q_store_examples.tinyllama_react_training"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found in project root${NC}"
    echo "   Create a .env file with your API keys:"
    echo "   PINECONE_API_KEY=your_key_here"
    echo "   PINECONE_ENVIRONMENT=us-east-1"
    echo "   IONQ_API_KEY=your_key_here  # Optional"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

python -m q_store_examples.tinyllama_react_training

echo ""
echo "======================================================================"
echo -e "${GREEN}✅ Training workflow complete!${NC}"
echo "======================================================================"
echo ""
echo "📁 Output locations:"
echo "   Dataset: ./react_train.jsonl"
echo "   Model: ./tinyllama-react-quantum/"
echo ""
echo "📚 Next steps:"
echo "   - Test the fine-tuned model"
echo "   - Evaluate on React coding tasks"
echo "   - Iterate with more data"
echo ""
