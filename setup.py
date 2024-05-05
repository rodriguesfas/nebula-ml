from setuptools import setup, find_packages

setup(
    name='NebulaML',
    version='0.0.1',
    author='Seu Nome',
    author_email='seu@email.com',
    description='NebulaML is a Python library for machine learning algorithms implementation, including LVQ and other models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rodriguesfas/nebula-ml',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    keywords='machine-learning lvq algorithms',
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'datasets'
    ],
    python_requires='>=3.7',
)