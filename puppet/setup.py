from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'puppet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        # package.xml
        (os.path.join('share', package_name), ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.py')),
        # Cấu hình
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # RViz configs (nếu có)
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        # URDF/Xacro
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # Meshes
        (os.path.join('share', package_name, 'meshes'), glob('meshes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dungx',
    maintainer_email='dungx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_states = puppet.joint_states_publisher:main',
            'odometry = puppet.odometry:main', 
            #'rpm_publisher = puppet.rpm_publisher:main',
            #'calib_wb = puppet.calib_wb:main',simple
            'controller = puppet.controller:main',
            'scan_merger = puppet.scan_merger_simple:main',
            'robot_pose = puppet.robot_pose_in_map:main',
            'auto_mapping = puppet.auto_mapping:main'
        ],
    },
)
