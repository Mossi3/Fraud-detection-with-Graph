#!/bin/bash
# Credit Card Fraud Detection System - Setup Script
# This script sets up the entire fraud detection system

echo "🛡️ Credit Card Fraud Detection System - Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION found"
        return 0
    else
        print_error "Python 3 is not installed. Please install Python 3.9+ first."
        return 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
        return 0
    else
        print_error "pip3 is not installed. Please install pip first."
        return 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip3 install -r requirements.txt
        if [ $? -eq 0 ]; then
            print_success "Dependencies installed successfully"
        else
            print_error "Failed to install dependencies"
            return 1
        fi
    else
        print_error "requirements.txt not found"
        return 1
    fi
}

# Generate sample data
generate_data() {
    print_status "Generating sample data..."
    
    if [ -f "data_processor.py" ]; then
        python3 data_processor.py
        if [ $? -eq 0 ]; then
            print_success "Sample data generated successfully"
        else
            print_warning "Failed to generate sample data (this is optional)"
        fi
    else
        print_warning "data_processor.py not found"
    fi
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data models graph visualization api notebooks
    print_success "Directories created"
}

# Make scripts executable
make_executable() {
    print_status "Making scripts executable..."
    
    if [ -f "test_api.py" ]; then
        chmod +x test_api.py
    fi
    
    if [ -f "presentation_demo.sh" ]; then
        chmod +x presentation_demo.sh
    fi
    
    print_success "Scripts made executable"
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    python3 -c "
import torch
import pandas as pd
import numpy as np
import fastapi
import streamlit
import plotly
import networkx
print('All required packages imported successfully')
" 2>/dev/null
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
        return 0
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Display next steps
show_next_steps() {
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start the API server:"
    echo "   ${BLUE}python3 api.py${NC}"
    echo ""
    echo "2. In another terminal, start the web interface:"
    echo "   ${BLUE}streamlit run app.py${NC}"
    echo ""
    echo "3. Test the system:"
    echo "   ${BLUE}python3 test_api.py${NC}"
    echo ""
    echo "4. Run the presentation demo:"
    echo "   ${BLUE}./presentation_demo.sh${NC}"
    echo ""
    echo "5. Access the web interface:"
    echo "   ${BLUE}http://localhost:8501${NC}"
    echo ""
    echo "6. View API documentation:"
    echo "   ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
    echo "📚 For more information, see README.md"
    echo ""
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    # Check prerequisites
    if ! check_python; then
        exit 1
    fi
    
    if ! check_pip; then
        exit 1
    fi
    
    # Create directories
    create_directories
    
    # Install dependencies
    if ! install_dependencies; then
        print_error "Setup failed during dependency installation"
        exit 1
    fi
    
    # Generate sample data
    generate_data
    
    # Make scripts executable
    make_executable
    
    # Test installation
    if ! test_installation; then
        print_warning "Installation test failed, but setup may still work"
    fi
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"