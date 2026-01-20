import os
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    project_name = "market-trend-identification"
    
    # Define directory structure
    structure = {
        project_name: {
            "data": {
                "raw": [],
                "processed": [],
                "synthetic": [],
                "external": []
            },
            "src": {
                "data": [],
                "features": [],
                "models": [],
                "visualization": [],
                "utils": []
            },
            "notebooks": [],
            "models": {
                "saved_models": [],
                "checkpoints": []
            },
            "results": {
                "figures": [],
                "reports": [],
                "metrics": []
            },
            "tests": [],
            "docs": [],
            "configs": [],
            "scripts": [],
            "api": [],
            "docker": []
        }
    }
    
    def create_dirs(base_path, structure_dict):
        """Recursively create directories"""
        for dir_name, sub_dirs in structure_dict.items():
            dir_path = os.path.join(base_path, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
            
            # Create empty __init__.py for Python packages
            if dir_name in ["src", "data", "features", "models", "visualization", "utils"]:
                init_file = os.path.join(dir_path, "__init__.py")
                with open(init_file, "w") as f:
                    f.write("# Package initialization\n")
                print(f"  Created: {init_file}")
            
            if isinstance(sub_dirs, dict):
                create_dirs(dir_path, sub_dirs)
            elif isinstance(sub_dirs, list):
                for sub_dir in sub_dirs:
                    sub_dir_path = os.path.join(dir_path, sub_dir)
                    os.makedirs(sub_dir_path, exist_ok=True)
                    print(f"Created: {sub_dir_path}")
    
    # Create the structure
    base_dir = "."
    create_dirs(base_dir, structure)
    
    # Create essential files
    files_to_create = {
        "README.md": "# Market Trend Identification\n\nProject for identifying market trends from chart images.",
        "requirements.txt": "# Project dependencies\n\nnumpy>=1.21.0\npandas>=1.3.0\nscikit-learn>=1.0.0\nmatplotlib>=3.5.0\nyfinance>=0.2.0\ntensorflow>=2.10.0\nopencv-python>=4.6.0\n",
        "setup.py": "from setuptools import setup, find_packages\n\nsetup(\n    name='market-trend-identification',\n    version='0.1.0',\n    packages=find_packages(),\n    install_requires=[\n        'numpy>=1.21.0',\n        'pandas>=1.3.0',\n        'scikit-learn>=1.0.0',\n        'matplotlib>=3.5.0',\n        'yfinance>=0.2.0',\n    ],\n)",
        ".gitignore": "# Python\n__pycache__/\n*.py[cod]\n*$py.class\n*.so\n.Python\nbuild/\ndevelop-eggs/\ndist/\ndownloads/\neggs/\n.eggs/\nlib/\nlib64/\nparts/\nsdist/\nvar/\nwheels/\n*.egg-info/\n.installed.cfg\n*.egg\n\n# IDE\n.vscode/\n.idea/\n*.swp\n*.swo\n*~\n\n# Data\n*.csv\n*.pkl\n*.h5\n*.npy\n\n# Images\n*.png\n*.jpg\n*.jpeg\n*.gif\n\n# Logs\nlogs/\n*.log\n\n# Virtual Environment\nvenv/\nenv/\n.env\n\n# OS\n.DS_Store\nThumbs.db\n",
        "configs/config.yaml": "# Configuration file\n\ndata:\n  window_size: 50\n  prediction_window: 10\n  threshold: 0.05\n  symbols:\n    - AAPL\n    - MSFT\n    - GOOGL\n    - AMZN\n    - TSLA\n\nmodel:\n  test_size: 0.2\n  validation_size: 0.15\n  random_state: 42\n  n_features: 50\n\npaths:\n  data_dir: ./data\n  models_dir: ./models\n  results_dir: ./results\n",
        "src/__init__.py": "# Market Trend Identification Package\n\n__version__ = '0.1.0'\n",
        "src/data/__init__.py": "# Data collection and processing modules\n",
        "src/data/data_collector.py": "# Data collection implementation\n",
        "src/features/feature_extractor.py": "# Feature extraction implementation\n",
        "src/models/classifiers.py": "# Classifier implementations\n",
        "src/utils/helpers.py": "# Utility functions\n",
        "scripts/setup_environment.sh": "#!/bin/bash\n\n# Setup script for environment\necho 'Setting up environment...'\npip install -r requirements.txt\n",
        "scripts/run_pipeline.sh": "#!/bin/bash\n\n# Run complete pipeline\npython -m src.pipeline\n",
        "tests/test_data_collector.py": "# Tests for data collector\nimport unittest\n\nclass TestDataCollector(unittest.TestCase):\n    def test_sample(self):\n        self.assertTrue(True)\n\nif __name__ == '__main__':\n    unittest.main()\n",
        "docker/Dockerfile": "FROM python:3.9-slim\n\nWORKDIR /app\n\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\n\nCOPY . .\n\nCMD [\"python\", \"app.py\"]\n",
        "api/app.py": "# Flask API for predictions\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/predict', methods=['POST'])\ndef predict():\n    return jsonify({'status': 'API endpoint'})\n\nif __name__ == '__main__':\n    app.run(debug=True)\n"
    }
    
    for file_path, content in files_to_create.items():
        full_path = os.path.join(project_name, file_path) if not file_path.startswith(project_name + "/") else file_path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)
        print(f"Created file: {full_path}")
    
    print(f"\n✅ Project structure created at: {project_name}/")
    print("\nNext steps:")
    print(f"1. cd {project_name}")
    print("2. pip install -r requirements.txt")
    print("3. Run your notebooks or scripts")
    
    return project_name

if __name__ == "__main__":
    project_path = create_project_structure()