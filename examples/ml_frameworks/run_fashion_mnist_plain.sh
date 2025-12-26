#!/bin/bash
#
# Run script for Fashion MNIST Plain Python Example
#
# Usage:
#   ./run_fashion_mnist_plain.sh           # Mock mode with defaults
#   ./run_fashion_mnist_plain.sh --real    # Real quantum hardware
#   ./run_fashion_mnist_plain.sh --help    # Show help
#

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default parameters
MODE="mock"
SAMPLES=100        # Limit test samples for faster execution
BATCH_SIZE=8       # Small batch size for testing
SKIP_CONFIRM=false # Skip confirmation prompt
PYTHON_SCRIPT="$SCRIPT_DIR/fashion_mnist_plain.py"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Parse arguments
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run Fashion MNIST Plain Python example with Q-Store v4.1

OPTIONS:
    --real              Use real IonQ quantum hardware (requires IONQ_API_KEY in .env)
    --mock              Use mock simulator backend (default)
    --samples N         Number of test samples to process (default: 100)
    --batch-size N      Batch size for inference (default: 8)
    -y, --yes           Skip confirmation prompt for real hardware
    -h, --help          Show this help message

EXAMPLES:
    # Run with mock backend (fast, no API keys needed)
    $(basename "$0")

    # Run with real quantum hardware (requires API keys)
    $(basename "$0") --real

    # Run with more samples
    $(basename "$0") --samples 500 --batch-size 16

    # Run with real hardware and custom parameters
    $(basename "$0") --real --samples 50 --batch-size 4

ENVIRONMENT:
    Create examples/.env with:
    - IONQ_API_KEY: Your IonQ API key (required for --real)
    - IONQ_TARGET: simulator or qpu.harmony (optional)

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --real)
            MODE="real"
            shift
            ;;
        --mock)
            MODE="mock"
            shift
            ;;
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -y|--yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
PYTHON_CMD="python"
SCRIPT_ARGS="--samples $SAMPLES --batch-size $BATCH_SIZE"

if [ "$MODE" = "real" ]; then
    SCRIPT_ARGS="--no-mock $SCRIPT_ARGS"
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo -e "${YELLOW}Consider activating your virtual environment first:${NC}"
    echo -e "${YELLOW}  source test_env/bin/activate${NC}"
    echo ""
fi

# Check if script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}Error: Script not found: $PYTHON_SCRIPT${NC}"
    exit 1
fi

# Check for .env file if using real hardware
if [ "$MODE" = "real" ]; then
    ENV_FILE="$PROJECT_ROOT/examples/.env"
    if [ ! -f "$ENV_FILE" ]; then
        echo -e "${RED}Error: .env file not found at $ENV_FILE${NC}"
        echo -e "${YELLOW}Create it from examples/.env.example and set IONQ_API_KEY${NC}"
        exit 1
    fi

    # Check if IONQ_API_KEY is set
    if ! grep -q "IONQ_API_KEY=" "$ENV_FILE" 2>/dev/null; then
        echo -e "${RED}Error: IONQ_API_KEY not found in $ENV_FILE${NC}"
        exit 1
    fi
fi

# Print configuration
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Fashion MNIST Plain Python - Q-Store v4.1${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Mode: ${YELLOW}$MODE${NC}"
echo -e "  Test samples: ${YELLOW}$SAMPLES${NC}"
echo -e "  Batch size: ${YELLOW}$BATCH_SIZE${NC}"
echo -e "  Script: ${YELLOW}$(basename $PYTHON_SCRIPT)${NC}"
echo ""

if [ "$MODE" = "real" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Using real quantum hardware${NC}"
    echo -e "${YELLOW}   This will consume IonQ API credits${NC}"
    echo ""
    if [ "$SKIP_CONFIRM" = "false" ]; then
        read -p "Continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    else
        echo -e "${GREEN}Auto-confirmed (--yes flag)${NC}"
        echo ""
    fi
fi

# Run the script
echo -e "${GREEN}Starting execution...${NC}"
echo ""
echo -e "${YELLOW}Command:${NC} $PYTHON_CMD $PYTHON_SCRIPT $SCRIPT_ARGS"
echo ""

cd "$PROJECT_ROOT"
$PYTHON_CMD "$PYTHON_SCRIPT" $SCRIPT_ARGS

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Execution completed successfully${NC}"
else
    echo -e "${RED}✗ Execution failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
