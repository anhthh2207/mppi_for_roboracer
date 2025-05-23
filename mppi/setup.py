from setuptools import setup

package_name = 'mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Anh Tran',
    maintainer_email='anhthh@seas.upenn.edu',
    description='f1tenth mppi',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mppi_node = mppi.mppi_node:main',
        ],
    },
)
