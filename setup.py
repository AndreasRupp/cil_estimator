import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setuptools.setup(
  name='ecdf_estimator',
  version='0.1.3',
  author='Andreas Rupp',
  author_email='info@rupp.ink',
  description='Python package for parameter estimation of random data',
  long_description=long_description,
  long_description_content_type="text/markdown",
  url='https://github.com/AndreasRupp/ecdf_estimator',
  project_urls = {
    "Bug Tracker": "https://github.com/AndreasRupp/ecdf_estimator/issues"
  },
  license='LGPL-2.1',
  packages=['ecdf_estimator'],
  install_requires=[
    'requests',
    'numpy>=1.19.5',
    'matplotlib>=3.6'
  ],
)
