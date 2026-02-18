from setuptools import setup, find_packages
from typing import List
import os

def get_requirements(file_path:str)->List[str]:
    '''this funciton return the list of requirements'''

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements
setup(
    name = "FirstML",
    version="0.1.0",
    packages=find_packages(),
    author="Dushyanth",
    install_requires=get_requirements("requirements.txt"),
)