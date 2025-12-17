#!/bin/bash
# release.sh
# Creates a release tag, pushes to remote, and triggers the build-wheels workflow

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Q-Store Release Script"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "src/q_store" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Check if version argument is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Version number required${NC}"
    echo ""
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 3.4.0"
    echo ""
    exit 1
fi

VERSION=$1
TAG="v${VERSION}"

# Validate version format (basic check)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Invalid version format${NC}"
    echo "Version should be in format: X.Y.Z (e.g., 3.4.0)"
    exit 1
fi

echo -e "${BLUE}Preparing release: ${TAG}${NC}"
echo ""

# Check if tag already exists locally
if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Tag ${TAG} already exists locally${NC}"
    read -p "Do you want to delete and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git tag -d "$TAG"
        echo -e "${GREEN}  ✓ Local tag deleted${NC}"
    else
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo -e "${BLUE}Current branch: ${CURRENT_BRANCH}${NC}"
echo ""

# Confirm release
echo -e "${YELLOW}You are about to:${NC}"
echo "  1. Create tag: ${TAG}"
echo "  2. Push tag to origin"
echo "  3. Trigger build-wheels workflow on GitHub Actions"
echo ""
read -p "Continue with release? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}Aborted${NC}"
    exit 1
fi
echo ""

# Step 1: Create the tag
echo -e "${GREEN}Step 1: Creating release tag...${NC}"
git tag -a "$TAG" -m "Release ${TAG}"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Tag ${TAG} created${NC}"
else
    echo -e "${RED}  ✗ Failed to create tag${NC}"
    exit 1
fi
echo ""

# Step 2: Push the code to remote
echo -e "${GREEN}Step 2: Pushing code to origin...${NC}"
git push origin "$CURRENT_BRANCH"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Code pushed to origin/${CURRENT_BRANCH}${NC}"
else
    echo -e "${RED}  ✗ Failed to push code${NC}"
    echo -e "${YELLOW}  Rolling back tag...${NC}"
    git tag -d "$TAG"
    exit 1
fi
echo ""

# Step 3: Push the tag to trigger workflow
echo -e "${GREEN}Step 3: Pushing release tag...${NC}"
git push origin "$TAG"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}  ✓ Tag pushed to origin${NC}"
else
    echo -e "${RED}  ✗ Failed to push tag${NC}"
    echo -e "${YELLOW}  Note: Local tag exists, you may need to delete it manually${NC}"
    exit 1
fi
echo ""

# Success message
echo -e "${GREEN}=========================================="
echo "Release ${TAG} Complete!"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}What happens next:${NC}"
echo "  1. GitHub Actions will automatically start the build-wheels workflow"
echo "  2. Wheels will be built for:"
echo "     - Linux (x86_64)"
echo "     - macOS (x86_64 and ARM64)"
echo "     - Windows (AMD64)"
echo "  3. Check workflow progress at:"
echo "     https://github.com/YOUR_USERNAME/q-store/actions"
echo ""
echo -e "${YELLOW}To monitor the build:${NC}"
echo "  gh run list --workflow=build-wheels.yml"
echo "  gh run watch"
echo ""
echo -e "${YELLOW}To download artifacts after build completes:${NC}"
echo "  gh run download"
echo ""
