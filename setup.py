from setuptools import setup, find_packages
setup(
    name='streamsculptor',
    version='0.0',
    packages=find_packages(),  # Automatically find packages, including subdirectories
    package_data={
        'streamsculptor': [
            'data/LMC_MW_potential/*',  
            'examples/*',  
        ],
    },
    include_package_data=True,  # Ensure that non-Python files are included
)