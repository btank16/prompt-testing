#!/bin/bash

# Navigate to the project directory (where this script lives)
cd "$(dirname "$0")"

echo "=== LLM Prompt Tester ==="
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
    echo ""
fi

# Activate venv
source venv/bin/activate

# Install/update requirements
echo "Checking dependencies..."
pip install -q -r requirements.txt
echo "Dependencies ready."
echo ""

# Launch Streamlit
echo "Launching Streamlit app..."
echo "Press Ctrl+C to stop."
echo ""
streamlit run app.py
