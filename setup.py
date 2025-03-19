from setuptools import find_packages,setup
from typing import List

HYPERN_E = "-e ."
def get_requirement(file_path:str) -> List[str]:

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if HYPERN_E in requirements:
            requirements.remove(HYPERN_E)
    
    return requirements

setup(
    name="MLProject",
    version='1.0.0',
    author='SJangid',
    author_email='shantu88jangid@gmail.com',
    packages=find_packages(),
    install_requires=get_requirement('requirements.txt')
)