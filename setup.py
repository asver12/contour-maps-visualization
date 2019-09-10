from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="contour_visualization",
      version="1.0.0",
      description="Contour-maps-visualisation",
      author="Laines Schmalwasser",
      license="2019 Laines Schmalwasser",
      packages=find_packages(),
      requirements=requirements
      )
