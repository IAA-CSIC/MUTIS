import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mutis",
    version="0.0.1",
    author="Juan Escudero & Jose Enrique Ruiz",
    author_email="jescudero@iaa.es, jer@iaa.es",
    description="A Python package for muti-wavelength time series analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bultako/mutis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)