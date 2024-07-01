from pathlib import Path
from setuptools import setup, find_packages
# TODO: - ADD readme.md,
#       - fix this setup (remove DocumentAI),
#       - setup the version,
#       - But before all this implement FEBDS algorithm,
#       - Add what's new between version in documentation
#       - Add License
with open("requirements.txt") as f:
    required = f.read().splitlines()

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="DocumentAI_std",
    version="0.2.8.dev1",
    packages=find_packages(exclude=["DocumentAI_std.tests"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    url="https://github.com/LATIS-DocumentAI-Group/medical-image-std",
    license="MIT",
    author="Hamza Gbada",
    author_email="hamza.gbada@gmail.com, karim.kalti@fsm.rnu.tn",
    python_requires=">=3.11, <3.13",
    description="The main standards for Latis medical images processing projects",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
