from setuptools import setup, find_packages

setup(
    name="hallucinationguard",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-dotenv"
    ],
    author="Sathvika",
    description="Confidence scoring and hallucination detection for LLM pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8"
)