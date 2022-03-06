import setuptools
import pkg_resources
import pathlib
import io
import os

NAME = "hbmedicalprocessing"
VERSION = "0.0.1"

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(os.path.join('source',NAME, 'requirements.txt'), "r", encoding="utf-8") as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]


setuptools.setup(
    name=NAME,
    version=VERSION,

    author="Raphael Baumann",
    author_email="raphael.baumann@st.oth-regensburg.de",

    description="Package for Classyfying House Brackmann scores and the including seperate Modules",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://re-mic.de/",

    project_urls={
        "Regensburg Medical Image Cumputing": "https://re-mic.de/",
        "Ostbayerische Technische Hochschule Regensburg": "https://www.oth-regensburg.de/",
        "Face-Alignment-Framework Source": "https://github.com/1adrianb/face-alignment",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: JavaScript",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: FastAPI",
        "Framework :: Jupyter :: JupyterLab",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: System :: Distributed Computing",

    ],

    install_requires=install_requires,
    license='GNU GPL v3',
    package_dir={"": "source"},
    packages=setuptools.find_packages(where="source"),
    package_data = {
        # If any package contains *.txt, ... files, include them:
        '': ['*.txt', '*.md', '*.yaml', '*.ipynb', "../../*.md", "./models/*.yaml",
             "./static/js/*.js", "./static/css/*.css", "./static/templates/*.html",],
    },
    #include_package_data=True,
    python_requires=">=3.6",
)
