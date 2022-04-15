from setuptools import setup

setup(
    name='emulator_utils',
    version='0.0.0',    
    description='Python package with emulation utilties ',
    url='https://github.com/patricialarsen/emulator_utils',
    author='Patricia Larsen',
    author_email='prlarsen@anl.gov',
    license='BSD 3-clause',
    packages=['emulator_utils'],
    install_requires=[
                      'numpy', 
                      'pygio',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)

