from setuptools import setup, find_packages

setup(
    name='pathwise_grad_kumar',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
