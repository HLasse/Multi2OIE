import setuptools

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setuptools.setup(
    name="relationextraction",
    version="0.0.5",
    author="Lasse Hansen",
    author_email="lasseh0310@gmail.com",
    description="A library for extracting relations within a text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HLasse/Multi2OIE",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0,<1.12.0",
        "numpy>=1.19.5,<1.24.0",
        "pandas>=1.1.5,<1.5.0",
        "spacy>=3.2.0,<3.3.0",
        "transformers>=4.11.3,<4.18.0",
        "spacy-transformers>=1.1.2,<1.2.0",
    ],
    keywords=["NLP", "knowledge graphs", "relation extraction", "triplets"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Windoiws 10",
        "Environment :: GPU :: NVIDIA CUDA :: 10.0",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
    ],
)
