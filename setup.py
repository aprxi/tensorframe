import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorframe",
    version="0.80.6",
    author="Anthony Potappel",
    author_email="anthony.potappel@gmail.com",
    description="Post alpha build",
    long_description="Post alpha build",
    long_description_content_type="text/markdown",
    url="https://tensorframe.ai",
    packages=setuptools.find_packages(),
    package_data={'tensorframe': ['core/libs/*.so']},
    zip_safe=False,
    #python_requires='==3.5.*, ==3.6.*, ==3.7.*',
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
