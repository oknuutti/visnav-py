# visnav-py
Visual Navigation Algorithms and Test Framework

## Installation
Needs:
* Python =3.5, numpy, numpy-quaternion, scipy, scikit-learn, astropy, opencv, pyopengl, qt, pyqt
* [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit)

At least on Windows, to get necessary Python packages it's easiest to use [Anaconda](https://www.continuum.io/downloads)

After installing Anaconda, run from command prompt:
* `conda create -n visnav pip`
* `activate visnav`
* `conda install -c conda-forge numpy scipy scikit-learn astropy opencv pyopengl qt pyqt`
* `pip install numpy-quaternion`

Download data files from my [Google Drive folder](https://drive.google.com/drive/folders/0ByfhOdRO_959X05jTWczWGxLUkk?usp=sharing)
into `data/` folder

To run standalone GUI mode:<br/>
`python src\visnav.py`

To run a Monte Carlo batch, open batch1.py in an editor to see what you are going to run, then:<br/>
`python src\batch1.py`

You also might want to look at `src/settings.py`.

## Documentation
This work started as a project done at Aalto University, School of Electrical Engineering.
The documentation done towards those credits can be found [here](https://docs.google.com/document/d/1lXqXdR02dAcGPsClwZOXj39RbBfrcscxIKrUyMY_WGU/edit#heading=h.dw2dac9r7xzm).

I find `hg` easier to use than `git`, so for this repo I've used `hg` with `hggit` extension.
Seems that it didn't change `.hgignore` file into `.gitignore`.
