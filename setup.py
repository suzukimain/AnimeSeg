from setuptools import setup, find_packages

setup(
    name="anime_seg",
    version="0.1.0",
    description="Anime Character Segmentation with DINOv2",
    long_description=open("README.md").read() if open("README.md").read() else "",
    long_description_content_type="text/markdown",
    author="ZetaX",
    author_email="info@zetax.jp",
    url="https://github.com/suzukimain/AnimeSeg",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "Pillow",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "peft",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
