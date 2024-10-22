from setuptools import setup, find_packages

setup(
    name="edu_irt",
    version="0.1.1",
    packages=find_packages(),
    description="created an IRT model for estimating item parameters",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/glassesholder/edu_irt.git",
    author="Hyojun LEE",
    author_email="statisticlee@gmail.com",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[    # 필요한 외부 패키지
        "numpy>=2.1.2",  
        "pandas>=2.2.3",
        "scipy>=1.14.1",
        "matplotlib>=3.9.2"
    ],
)