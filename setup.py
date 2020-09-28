import re

__author__ = "Olli Knuuttila"
__date__ = "$Nov 25, 2016 8:05:11 PM$"

from setuptools import setup, find_packages


# with open('../README.md') as f:
# 	long_description = f.read()
# 	long_description = re.sub(r'</?div[^>]*>|\r', '', long_description, flags=re.M)


setup(
    name='visnav',
    version='0.1',
    packages=find_packages(include=['visnav*']),
    include_package_data=True,
    package_data={'visnav.render': ['*.frag', '*.vert', '*.geom']},

    # Declare your packages' dependencies here, for eg:
    install_requires=['moderngl>=5.5.0', 'moderngl.ext.obj>=1.0.0', 'scipy', 'numpy', 'astropy', 'numba=>0.39',
                      # 'opencv', 'quaternion'  --these didn't work, conda versions were not noticed by pip
                      ],

    author=__author__,
    author_email='olli.knuuttila@gmail.com',

    summary='Visual navigation around solar-system small bodies',
    url='https://github.com/oknuutti/visnav-py',
    license='MIT',
)