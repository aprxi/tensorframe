import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorframe",
    version="0.80.0",
    author="Anthony Potappel",
    author_email="anthony.potappel@gmail.com",
    description="Post alpha build",
    long_description="Post alpha build",
    long_description_content_type="text/markdown",
    url="https://tensorframe.ai",
    packages=setuptools.find_packages(),
    package_data={'tensorframe': ['core/libs/*.so']},
    zip_safe=False,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: LINUX",
    ),
)
