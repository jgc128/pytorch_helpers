from setuptools import setup, find_packages

setup(
    name="pytorch_helpers",
    version="0.1.0",
    packages=find_packages(),
    install_requires=['numpy', 'opencv-python', 'imgaug'],
    author="Alexey Romanov",
    author_email="aromanov@cs.uml.edu",
    description="Different helpers for building PyTorch models",
    url="https://github.com/jgc128/pytorch_helpers",
)
