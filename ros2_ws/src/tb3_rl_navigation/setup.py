from setuptools import setup
import os
from glob import glob

package_name = 'tb3_rl_navigation'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Saigirish S',
    maintainer_email='saigirish@todo.todo',
    description='Lightweight ROS 2 deployment and inference package for TurtleBot3 LiDAR RL navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deploy_node = tb3_rl_navigation.deploy_node:main',
        ],
    },
)

