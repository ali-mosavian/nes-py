"""The setup script for installing and distributing the nes-py package."""
import os
import subprocess
from glob import glob
from typing import List
from pathlib import Path

import pybind11
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_ext import build_ext
from pybind11.setup_helpers import Pybind11Extension


class MakeBuilder(build_ext):
    """Custom builder that uses Make."""
    
    def build_extension(self, ext: Pybind11Extension) -> None:
        """Build the extension using Make."""
        extension_path: str = os.path.dirname(ext.sources[0])
        make_path: str = os.path.join(extension_path, '..')
        
        try:
            n_jobs = min(16, os.cpu_count())
            subprocess.check_call(['make', '-C', make_path, f'-j{n_jobs}'])
            
            built_lib: str = os.path.join(make_path, 'libemulator.so')
            target_lib: str = self.get_ext_fullpath(ext.name)
            
            if os.path.exists(built_lib):
                os.makedirs(os.path.dirname(target_lib), exist_ok=True)
                import shutil
                shutil.copy2(built_lib, target_lib)
            else:
                raise RuntimeError(f'Built library not found at {built_lib}')
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Error building extension: {e}')


def read_readme() -> str:
    """Read the README.md file and return its content."""
    with open('README.md') as f:
        return f.read()


def get_source_files() -> List[str]:
    """Get all C++ source files needed for compilation."""
    return glob('nes_py/nes/src/*.cpp') + glob('nes_py/nes/src/mappers/*.cpp')


def configure_compiler() -> None:
    """Configure the C++ compiler settings."""
    os.environ['CC'] = 'g++'
    os.environ['CCX'] = 'g++'


def get_extension_modules() -> List[Pybind11Extension]:
    """Create and return the extension modules configuration."""
    sources: List[str] = get_source_files()
    
    return [
        Pybind11Extension(
            name='nes_py.emulator',
            sources=sources,
            include_dirs=[
                'nes_py/nes/include',
                'nes_py/nes/src',
                pybind11.get_include(),
                pybind11.get_include(user=True)
            ],
            cxx_std=14,
            extra_compile_args=['-O3', '-Wall', '-Wextra', '-pedantic'],
        ),
    ]


def get_requirements() -> List[str]:
    """Get the requirements for the package."""    
    with open('requirements.txt') as f:
        return [line.strip() for line in f]


def main() -> None:
    """Main setup configuration."""
    configure_compiler()
    
    setup(
        name='nes_py',
        version='9.2.0',
        description='An NES Emulator with Gymnasium interface',
        long_description=read_readme(),
        long_description_content_type='text/markdown',
        keywords='NES Emulator, Gymnasium',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: MacOS :: MacOS X',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3 :: Only',
            *[f'Programming Language :: Python :: 3.{v}' for v in range(8, 13)],
            'Topic :: Games/Entertainment',
            'Topic :: Software Development :: Libraries :: Python Modules',
            'Topic :: System :: Emulators',
        ],
        url='https://github.com/Kautenja/nes-py',
        author='Christian Kauten',
        author_email='kautencreations@gmail.com',
        license='MIT',
        packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
        package_data={'nes_py': ['../requirements.txt']},
        ext_modules=get_extension_modules(),
        zip_safe=False,
        install_requires=get_requirements(),
        extras_require={
            'dev': [
                'pytest>=7.0.0',
                'pytest-cov>=4.0.0',
                'black>=23.0.0',
                'mypy>=1.0.0',
            ],
        },
        entry_points={
            'console_scripts': [
                'nes_py=nes_py.app.cli:main',
            ],
        },
        python_requires='>=3.8',
        cmdclass={'build_ext': MakeBuilder},
    )


if __name__ == '__main__':
    main()
