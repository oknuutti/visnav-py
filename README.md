# visnav-py
Test Framework for Visual Navigation Algorithms

## Installation
Needs:
* Python >3.3, pyqt5, scipy, quaternion, astropy, opencv3
* [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit)

At least on Windows, to get necessary Python packages it's easiest to use [Anaconda](https://www.continuum.io/downloads)

After installing Anaconda, run from command prompt:
* `conda install -c conda-forge opencv`
* `conda install -c moble quaternion`

Download data files from my [Google Drive folder](https://drive.google.com/drive/folders/0ByfhOdRO_959X05jTWczWGxLUkk?usp=sharing)
into `data/` folder

To run standalone GUI mode in Windows:<br/>
`"C:\Program Files\Anaconda3\python" src\visnav.py`

To run a Monte Carlo batch, open batch1.py in an editor to see what you are going to run, then:<br/>
`"C:\Program Files\Anaconda3\python" src\batch1.py`

You also might want to look at `src/settings.py`.

## Documentation
This work started as a project done at Aalto University, School of Electrical Engineering.
The documentation done towards those credits can be found [here](https://docs.google.com/document/d/1lXqXdR02dAcGPsClwZOXj39RbBfrcscxIKrUyMY_WGU/edit#heading=h.dw2dac9r7xzm).

I find `hg` easier to use than `git`, so for this repo I've used `hg` with `hggit` extension.
Seems that it didn't change `.hgignore` file into `.gitignore`.
