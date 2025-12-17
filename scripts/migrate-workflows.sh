#!/bin/bash
# Migration Script for GitHub Actions Workflow Improvements
# This script helps migrate from old separate workflows to the new unified workflow

set -e

echo "=================================================="
echo "GitHub Actions Workflow Migration Script"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: Not in a git repository${NC}"
    echo "Please run this script from the root of your git repository"
    exit 1
fi

echo "Step 1: Backing up existing workflows..."
mkdir -p .github/workflows/old-workflows-backup

if [ -d ".github/workflows" ]; then
    # Count existing workflow files
    workflow_count=$(find .github/workflows -maxdepth 1 -name "*.yml" -o -name "*.yaml" | wc -l)
    
    if [ "$workflow_count" -gt 0 ]; then
        echo "Found $workflow_count existing workflow files"
        
        # Backup existing workflows
        for file in .github/workflows/*.yml .github/workflows/*.yaml 2>/dev/null; do
            if [ -f "$file" ]; then
                filename=$(basename "$file")
                cp "$file" ".github/workflows/old-workflows-backup/$filename"
                echo "  ✓ Backed up: $filename"
            fi
        done
        
        echo -e "${GREEN}✓ Backup completed${NC}"
        echo "  Location: .github/workflows/old-workflows-backup/"
    else
        echo -e "${YELLOW}No existing workflows found${NC}"
    fi
else
    mkdir -p .github/workflows
    echo "Created .github/workflows directory"
fi

echo ""
echo "Step 2: Checking for required files..."

# Check for package configuration
if [ -f "setup.py" ]; then
    echo -e "  ${GREEN}✓ Found setup.py${NC}"
    has_config=true
elif [ -f "pyproject.toml" ]; then
    echo -e "  ${GREEN}✓ Found pyproject.toml${NC}"
    has_config=true
else
    echo -e "  ${RED}✗ No setup.py or pyproject.toml found${NC}"
    echo "    You need one of these files to build wheels"
    has_config=false
fi

# Check for __init__.py
if find . -name "__init__.py" | grep -q .; then
    echo -e "  ${GREEN}✓ Found __init__.py${NC}"
else
    echo -e "  ${YELLOW}⚠ No __init__.py found${NC}"
    echo "    Make sure your package has proper Python module structure"
fi

# Check for README
if [ -f "README.md" ] || [ -f "README.rst" ] || [ -f "README.txt" ]; then
    echo -e "  ${GREEN}✓ Found README${NC}"
else
    echo -e "  ${YELLOW}⚠ No README found${NC}"
    echo "    PyPI packages should have a README"
fi

# Check for LICENSE
if [ -f "LICENSE" ] || [ -f "LICENSE.txt" ] || [ -f "LICENSE.md" ]; then
    echo -e "  ${GREEN}✓ Found LICENSE${NC}"
else
    echo -e "  ${YELLOW}⚠ No LICENSE found${NC}"
    echo "    Open source packages should have a LICENSE file"
fi

if [ "$has_config" = false ]; then
    echo ""
    echo -e "${RED}Cannot continue without setup.py or pyproject.toml${NC}"
    exit 1
fi

echo ""
echo "Step 3: Installing new workflow files..."

# Check if new workflow files are in current directory
new_workflows=()
if [ -f "build-wheels-improved.yml" ]; then
    new_workflows+=("build-wheels-improved.yml")
fi
if [ -f "build-wheels-test.yml" ]; then
    new_workflows+=("build-wheels-test.yml")
fi

if [ ${#new_workflows[@]} -eq 0 ]; then
    echo -e "${YELLOW}New workflow files not found in current directory${NC}"
    echo "Please ensure build-wheels-improved.yml and build-wheels-test.yml are present"
    echo ""
    echo "You can:"
    echo "  1. Download them from the repository"
    echo "  2. Copy them to this directory"
    echo "  3. Re-run this script"
    exit 1
fi

for workflow in "${new_workflows[@]}"; do
    cp "$workflow" ".github/workflows/$workflow"
    echo -e "  ${GREEN}✓ Installed: $workflow${NC}"
done

echo ""
echo "Step 4: Checking GitHub secrets..."
echo ""
echo "Please verify the following secrets are set in your GitHub repository:"
echo "  Settings → Secrets and variables → Actions"
echo ""
echo "  Required:"
echo "    • PYPI_API_TOKEN - Your PyPI API token"
echo "      Get from: https://pypi.org/manage/account/token/"
echo ""
echo -e "${YELLOW}Press Enter when you've verified the secret is set...${NC}"
read

echo ""
echo "Step 5: Creating test commit..."
echo ""

git add .github/workflows/

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo "Changes staged:"
    git diff --cached --stat
    echo ""
    echo -e "${YELLOW}Ready to commit? (y/n)${NC}"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        git commit -m "chore: upgrade GitHub Actions workflows

- Add unified build-wheels-improved.yml with pre-checks
- Add build-wheels-test.yml for testing without publishing
- Backup old workflows to old-workflows-backup/
- Improve error handling and publishing coordination"
        
        echo -e "${GREEN}✓ Changes committed${NC}"
        
        echo ""
        echo "Step 6: Ready to push?"
        echo "Run: git push origin $(git branch --show-current)"
    else
        echo "Commit skipped. You can commit manually later."
    fi
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Migration Complete!${NC}"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Test the new workflows:"
echo "   • Push to a branch and open a PR to trigger test workflow"
echo "   • Or manually trigger 'Test Build Wheels' workflow"
echo ""
echo "2. Create a test release:"
echo "   git tag v1.0.0-test.1"
echo "   git push origin v1.0.0-test.1"
echo ""
echo "3. Monitor the workflow:"
echo "   • Go to GitHub Actions tab"
echo "   • Watch 'Build and Publish Wheels' workflow"
echo ""
echo "4. If successful, clean up old workflows:"
echo "   git rm .github/workflows/old-workflows-backup/*"
echo "   git commit -m 'chore: remove old workflow backups'"
echo ""
echo "5. Read the documentation:"
echo "   • WORKFLOW_IMPROVEMENTS.md - Complete guide"
echo "   • Troubleshooting section for common issues"
echo ""
echo "=================================================="
