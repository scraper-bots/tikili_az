"""
Setup script for DeepSeek AZE package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read version
version = "0.1.0"

setup(
    name="deepseek-aze",
    version=version,
    author="Ismat Samadov",
    author_email="ismat.samadov@gmail.com",
    description="Fine-tuning DeepSeek Large Language Models for Azerbaijani Language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ismat-Samadov/deepseek_AZE",
    packages=find_packages(include=["src*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "pylint>=2.17.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
        ],
        "evaluation": [
            "evaluate>=0.4.0",
            "rouge-score>=0.1.2",
            "sacrebleu>=2.3.1",
            "bert-score>=0.3.13",
        ],
        "api": [
            "fastapi>=0.103.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.26.0",
            "gradio>=3.44.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "deepseek-aze-train=train:main",
            "deepseek-aze-evaluate=evaluate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "configs/**/*.yaml"],
    },
    keywords=[
        "azerbaijani",
        "nlp",
        "language-model",
        "fine-tuning",
        "deepseek",
        "lora",
        "qlora",
        "transformers",
        "pytorch",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Ismat-Samadov/deepseek_AZE/issues",
        "Source": "https://github.com/Ismat-Samadov/deepseek_AZE",
        "Documentation": "https://github.com/Ismat-Samadov/deepseek_AZE#readme",
    },
)