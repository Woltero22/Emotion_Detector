from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emotion-detector",
    version="1.0.0",
    description="Real-time facial emotion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=requirements,
)
