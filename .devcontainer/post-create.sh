#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Setting up CLARA-SSoT development environment..."

# Git 설정 (이미 있으면 덮어쓰지 않기)
if ! git config --global user.name >/dev/null 2>&1; then
  git config --global user.name "Gibum Lee"
fi

if ! git config --global user.email >/dev/null 2>&1; then
  git config --global user.email "gibum@example.com"
fi

# Poetry 설치
echo "📦 Installing Poetry..."
curl -sSL https://install.python-poetry.org -o install-poetry.py
python3 install-poetry.py
rm install-poetry.py

# Poetry PATH
export PATH="$HOME/.local/bin:$PATH"
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Poetry 설정
poetry config virtualenvs.in-project true

# 의존성 설치
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing Python dependencies with Poetry..."
    poetry install
elif [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies with pip..."
    pip install -r requirements.txt
fi

# DVC 초기화 (poetry 환경 사용)
if [ ! -d ".dvc" ]; then
    echo "📊 Initializing DVC..."
    poetry run dvc init
    git add .dvc .dvcignore || true
fi

# Pre-commit 훅 설치 (poetry 환경 사용)
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    poetry run pre-commit install
fi

echo "✅ Development environment setup complete!"
