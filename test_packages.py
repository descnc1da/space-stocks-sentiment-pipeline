import importlib
import sys

names = ['vaderSentiment', 'torch', 'transformers', 'newsapi', 'dotenv', 'pandas']

for name in names:
    try:
        module = importlib.import_module(name)
        sys.modules[name] = module
        print(f"'{name}' successfully imported")
    except ImportError:
        print(f"can't find the '{name}' module")