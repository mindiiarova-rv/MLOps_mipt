import subprocess

# Запуск train.py
subprocess.run(["python", "ml_functional/train.py"])

# Запуск infer.py
subprocess.run(["python", "ml_functional/infer.py"])