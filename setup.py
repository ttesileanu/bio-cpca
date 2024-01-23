from distutils.core import setup

setup(
    name="ncpca",
    version="0.0.1",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    url="https://github.com/ttesileanu/bio-deep-ncpca",
    packages=["ncpca"],
    package_dir={"": "src"},
    install_requires=[
        "setuptools",
    ],
)
