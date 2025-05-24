from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="open-coscientist-agents",
    version="0.1.0",
    author="rwxai",
    author_email="",  # Add your email if you want to include it
    description="Implementation of multi-agent system for AI co-scientist",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conradry/open-coscientist-agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.1",
        "langgraph>=0.0.1",
        "openai>=1.0.0",
        "typing-extensions>=4.0.0",
        "ipython>=8.0.0",  # For notebook support
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "ruff>=0.0.1",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
) 