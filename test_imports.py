"""Quick import test to verify all dependencies installed correctly."""

import sys

def test_imports():
    errors = []
    libs = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("statsmodels", "statsmodels"),
        ("osmnx", "osmnx"),
        ("networkx", "networkx"),
        ("ortools", "ortools"),
        ("streamlit", "streamlit"),
        ("folium", "folium"),
        ("streamlit_folium", "streamlit_folium"),
        ("plotly", "plotly"),
    ]

    for name, module in libs:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"  OK  {name:20s} {version}")
        except ImportError as e:
            errors.append(name)
            print(f"  ERR {name:20s} FAILED: {e}")

    if errors:
        print(f"\n  FAIL: {len(errors)} libraries missing: {', '.join(errors)}")
        sys.exit(1)
    else:
        print(f"\n  All {len(libs)} libraries imported successfully.")
        sys.exit(0)

if __name__ == "__main__":
    print("BinSense — Dependency Check")
    print("-" * 40)
    test_imports()
