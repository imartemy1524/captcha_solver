from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()
with open("./requirements.txt") as req_file:
    requirements =[i.replace('\n', '').replace('\r', '') for i in req_file]


setup(
  name='captcha_solve_adapter',
  packages=['captcha_solve_adapter'],
  version='1.0',
  license='MIT',
  author='IMCorp',
#  description='Library to solve vk captcha async/sync.\nFree.\nHigh speed.',
  long_description_content_type='text/markdown',
  long_description=long_description,
  package_data={'captcha_solve_adapter': ['*.onnx']},
  author_email='imartemy1524@gmail.com',
  keywords=['solver', 'captcha', 'adapter', 'solve', 'AI', 'onnx'],
  install_requires=requirements,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
