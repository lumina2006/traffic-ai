"""
Setup script for Traffic AI package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="traffic-ai",
    version="1.0.0",
    author="Traffic AI Team",
    author_email="contact@traffic-ai.com",
    description="Intelligent Traffic Analysis System with AI-powered vehicle detection, tracking, and prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lumina2006/traffic-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
        "web": [
            "streamlit>=1.24.0",
            "dash>=2.11.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "traffic-ai=traffic_ai.cli:main",
            "traffic-detect=traffic_ai.detection.cli:main",
            "traffic-analyze=traffic_ai.analysis.cli:main",
            "traffic-predict=traffic_ai.prediction.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "traffic_ai": ["config/*.yaml", "models/*.pt"],
    },
    zip_safe=False,
    keywords="traffic ai computer-vision object-detection tracking prediction smart-city",
    project_urls={
        "Bug Reports": "https://github.com/lumina2006/traffic-ai/issues",
        "Source": "https://github.com/lumina2006/traffic-ai",
        "Documentation": "https://traffic-ai.readthedocs.io/",
        "Funding": "https://github.com/sponsors/lumina2006",
    },
)