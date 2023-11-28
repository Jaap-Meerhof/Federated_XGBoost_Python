import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='SFXGBoost',
    author='Jaap Meerhof',
    author_email='jaapmeerhof"at"proton.me',
    description='Simple Federated XGBoost implementation as descibed in the paper: Tian etal 2020 FederBoost',
    keywords='Federated, XGBoost, package, Membership Inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Jaap-Meerhof/Federated_XGBoost_Python',
    project_urls={
        'Documentation': 'https://github.com/Jaap-Meerhof/Federated_XGBoost_Python',
        'Bug Reports':
        'https://github.com/Jaap-Meerhof/Federated_XGBoost_Python/issues',
        'Source Code': 'https://github.com/Jaap-Meerhof/Federated_XGBoost_Python',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8.16', #ONLY TESTED ON 3.8.16 and 3.8.17
    # install_requires=['Pillow'],
    extras_require={
        'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
    # entry_points={
    #     'console_scripts': [  # This can provide executable scripts
    #         'run=examplepy:main',
    # You can execute `run` in bash to run `main()` in src/examplepy/__init__.py
    #     ],
    # },
)
