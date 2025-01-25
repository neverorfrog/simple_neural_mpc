from setuptools import find_packages, setup

setup(
    name="simple_neural_mpc",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(
        where="src",
        include=[
            "simple_neural_mpc",
            "simple_neural_mpc.*",
            "neural_model_identification",
            "neural_model_identification.*",
        ],
    ),
    install_requires=[
        "casadi",
        "matplotlib",
        "numpy",
        "omegaconf",
        "pyyaml",
        "pyqt5",
        "scipy",
        "torch",
        "tqdm",
        "scikit-build",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A neural network for trajectory tracking",
    url="https://github.com/neverorfrog/simple_neural_mpc.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
