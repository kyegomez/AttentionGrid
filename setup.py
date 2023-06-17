from setuptools import setup, find_packages

setup(
  name = 'AttentionGrid',
  packages = find_packages(exclude=['examples']),
  version = '0.0.1',
  license='APACHE',
  description = 'AttentionGrid - Library',
  author = 'Phil Wang',
  author_email = 'kye@apac.ai',
  url = 'https://github.com/kyegomez/AttentionGrid',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers'
    'attention'
    'large language models'
  ],
  install_requires=[
    'torch',
    'einops'
    'triton'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: APACHE License',
    'Programming Language :: Python :: 3.6',
  ],
)