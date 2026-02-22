#!/bin/bash

# Navigate to backend directory
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "⬇️  Installing/Updating dependencies..."
pip install -r requirements.txt

# Run the server
echo "🚀 Starting Backend Server on port 8000..."
python3 app.py
