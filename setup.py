from setuptools import setup, find_packages
#setup(name='streamsculptor',
#        version='0.0',
#        packages=['streamsculptor','streamsculptor.data.LMC_MW_potential','streamsculptor.examples'])
setup(
    name='streamsculptor',
    version='0.0',
    packages=find_packages(),  # Automatically find packages, including subdirectories
    package_data={
        'streamsculptor': [
            'data/LMC_MW_potential/*',  # Include all files under 'data' directory
            'examples/*',  # Include all files under 'examples' directory
        ],
    },
    include_package_data=True,  # Ensure that non-Python files are included
)