#!/bin/bash

echo "🚀 Starting ECG Classification Frontend..."

cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo "🌐 Starting React development server on http://localhost:3002"
npm run dev

