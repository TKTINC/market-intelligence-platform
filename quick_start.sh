#!/bin/bash

echo "🚀 Market Intelligence Platform - Quick Start"
echo "============================================="

# Check if .env exists
if [[ ! -f ".env" ]]; then
    echo "📋 Setting up environment..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your API keys before continuing"
    echo "   Required: OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY"
    read -p "Press Enter after updating .env file..."
fi

echo "🚀 Starting deployment..."
./scripts/automation_scripts.sh --environment local --test-type sanity

echo ""
echo "🎉 Deployment completed!"
echo "📱 Frontend: http://localhost:3001"
echo "🔌 API: http://localhost:8000"
echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
echo ""
echo "🧪 Run full tests: python3 scripts/comprehensive_tests.py --env local --test-type all"
