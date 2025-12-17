#!/bin/bash
# test_workflow_locally.sh
# Test GitHub Actions workflows locally using act

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Local GitHub Actions Workflow Testing"
echo "=========================================="
echo ""

# Check if act is installed
if ! command -v act &> /dev/null && [ ! -f "./bin/act" ]; then
    echo -e "${RED}Error: 'act' is not installed${NC}"
    echo ""
    echo "Install with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
    echo ""
    exit 1
fi

# Use local act if available
ACT_CMD="act"
if [ -f "./bin/act" ]; then
    ACT_CMD="./bin/act"
fi

echo -e "${BLUE}Available workflows:${NC}"
echo "  1. build-wheels.yml      - Build wheels for all platforms"
echo "  2. build-windows.yml     - Build Windows wheels only"
echo "  3. build-linux.yml       - Build Linux wheels only"
echo "  4. build-macos.yml       - Build macOS wheels only"
echo ""

# Show available actions
echo -e "${YELLOW}Testing options:${NC}"
echo "  • List workflows:     $0 list"
echo "  • Test build job:     $0 build"
echo "  • Dry run:            $0 dry-run"
echo "  • Test specific job:  $0 <workflow-file> <job-name>"
echo ""

case "${1:-help}" in
    list)
        echo -e "${GREEN}Listing all workflows and jobs:${NC}"
        echo ""
        $ACT_CMD -l
        ;;
    
    dry-run)
        echo -e "${GREEN}Performing dry run (no execution):${NC}"
        echo ""
        $ACT_CMD -n
        ;;
    
    build)
        WORKFLOW="${2:-.github/workflows/build-wheels.yml}"
        echo -e "${GREEN}Testing build workflow: ${WORKFLOW}${NC}"
        echo -e "${YELLOW}Note: This will use Docker and may take a while${NC}"
        echo ""
        
        # Run with minimal configuration for testing
        $ACT_CMD push -W "$WORKFLOW" \
            --container-architecture linux/amd64 \
            -P ubuntu-22.04=catthehacker/ubuntu:act-22.04 \
            --env CIBW_BUILD="cp311-*" \
            --env CIBW_SKIP="*-win32 *-manylinux_i686 *-musllinux_*"
        ;;
    
    test-syntax)
        echo -e "${GREEN}Testing workflow syntax:${NC}"
        echo ""
        for workflow in .github/workflows/*.yml; do
            echo -e "${BLUE}Checking: $workflow${NC}"
            $ACT_CMD -W "$workflow" -n --quiet && echo -e "${GREEN}  ✓ Valid${NC}" || echo -e "${RED}  ✗ Invalid${NC}"
        done
        ;;
    
    windows)
        echo -e "${GREEN}Testing Windows workflow (limited on Linux):${NC}"
        echo -e "${YELLOW}Note: Full Windows testing requires Windows or WSL${NC}"
        echo ""
        $ACT_CMD push -W .github/workflows/build-windows.yml -n
        ;;
    
    *)
        if [ -f ".github/workflows/$1" ]; then
            WORKFLOW=".github/workflows/$1"
            JOB="${2:-}"
            
            echo -e "${GREEN}Testing workflow: ${WORKFLOW}${NC}"
            if [ -n "$JOB" ]; then
                echo -e "${BLUE}Job: ${JOB}${NC}"
                $ACT_CMD push -W "$WORKFLOW" -j "$JOB"
            else
                $ACT_CMD push -W "$WORKFLOW"
            fi
        else
            echo -e "${YELLOW}Usage:${NC}"
            echo "  $0 list                    # List all workflows and jobs"
            echo "  $0 dry-run                 # Dry run (shows what would execute)"
            echo "  $0 test-syntax             # Validate workflow YAML syntax"
            echo "  $0 build [workflow.yml]    # Test build workflow"
            echo "  $0 windows                 # Test Windows workflow (dry run)"
            echo "  $0 <workflow.yml> [job]    # Test specific workflow/job"
            echo ""
            echo -e "${YELLOW}Examples:${NC}"
            echo "  $0 list"
            echo "  $0 test-syntax"
            echo "  $0 dry-run"
            echo "  $0 build"
            echo "  $0 build-wheels.yml build_wheels"
            echo ""
            echo -e "${YELLOW}Quick test (recommended):${NC}"
            echo "  # Validate syntax without running"
            echo "  $0 test-syntax"
            echo ""
            echo "  # See what would execute"
            echo "  $0 dry-run"
            echo ""
            echo -e "${BLUE}Note: Full builds require Docker and significant time/resources${NC}"
        fi
        ;;
esac
