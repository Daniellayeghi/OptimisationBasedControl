from distutils.core import setup

import sys
print(sys.version_info)
if sys.version_info == (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported')

print('${CMAKE_CURRENT_BINARY_DIR}')
setup(
    name='cmake_cpp_pybind11',
    version='${PACKAGE_VERSION}',
    packages=['mj_derivative'],
    package_dir={
        '': '${CMAKE_CURRENT_BINARY_DIR}'
    },
    package_data={
        '': ['mj_derivative.so']
    }
)
