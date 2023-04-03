from setuptools import setup, find_packages

REQUIREMENT_FILE_NAME = "requirements.txt"
REMOVE_PACKAGE = "-e ."


def get_requirement_list(requirement_file_name=REQUIREMENT_FILE_NAME) -> list:
    try:
        requirement_list = None
        with open(requirement_file_name) as requirement_file:
            requirement_list = [requirement.replace("\n", "") for requirement in requirement_file]
            requirement_list.remove(REMOVE_PACKAGE)
        return requirement_list
    except Exception as e:
        raise e


setup(
    name="ADMET_PAMPA_NCATS-classification",
    description="This is a Binary classification. Given a compound's SMILES string, predict whether it is has high permeability (1) or low-to-moderate permeability (0) in PAMPA assay.",
    author="Hamed Mehrabi",
    packages=find_packages(),
    install_requires=get_requirement_list()
)
