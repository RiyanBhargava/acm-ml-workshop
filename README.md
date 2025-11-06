# Project Name

Project Information

## Setup Guide

This website used mkdocs to generate a static site. Follow the steps below to setup the project on your local machine.

Ensure Python is installed on your machine, this project was built with Python 3.10 but it should work with Python >= 3.6.

Open the terminal/Command Line and follow the steps below.

```bash
python -m venv .venv
```

For Windows
    
```bash
.venv/Scripts/activate
```

For Unix OSes

```bash
source .venv/bin/activate
```

Install the required packages

```bash
pip install -r requirements.txt
mkdocs serve # this will deploy the website locally, make sure all your work is in the docs/ folder
```

When the the main branch is updated, a workflow will run to deploy the website to gh-pages branch. Ensure workflows are given all read and write permissions in settings else the action will fail. During development forks or other branches can be used, but the main branch should be used only when ready for deployment.

Deployment can be setup via GitHub pages pointing to the gh-pages branch. DNS will have to be setup.

Files to be changed:

- CNAME
- mkdocs.yml
- index.md

Images can be put in the docs/assets folder. Requirement.txt must be updated with any of the python requirements.