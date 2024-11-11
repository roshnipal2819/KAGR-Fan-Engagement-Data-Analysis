import subprocess

libraries = [
    "pandas",
    "numpy",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "plotly",
    "folium",
    "scipy",
    "joblib",
    "xgboost",
    "streamlit",
    "openai",
    "beautifulsoup4",
    "requests",
    "pillow",
    "python-dotenv",
]

for lib in libraries:
    try:
        result = subprocess.run(["pip", "show", lib], capture_output=True, text=True, check=True)
        version_line = next((line for line in result.stdout.splitlines() if line.startswith("Version:")), None)
        if version_line:
            version = version_line.split()[1]
            print(f"{lib}: {version}")
        else:
            print(f"{lib}: Version not available")
    except subprocess.CalledProcessError:
        print(f"{lib}: Not installed")
