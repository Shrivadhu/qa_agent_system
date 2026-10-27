from pathlib import Path
from setuptools import setup, find_packages

setup(
    name="qa_agent_system",
    version="1.0.0",
    description="QA Agent Post-Session Analysis System",
    author="Your Organization",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        line.split("#")[0].strip()
        for line in Path(__file__).with_name("requirements.txt").read_text().splitlines()
        if line.split("#")[0].strip()
    ],
    package_data={"config": ["config.yaml"]},
    include_package_data=True,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
