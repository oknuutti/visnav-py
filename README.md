# visnav-py
Visual Navigation Algorithms and Test Framework

## Installation
Clone the repository to a desired place, cd to it.

Dependencies are listed at `visnav.env`, which is a conda env file. At least on Windows, to get the necessary Python environment ready, it's easiest to use [Anaconda](https://www.continuum.io/downloads). After possibly installing Anaconda, run from command prompt:
* `conda env create -f=visnav.env`
* `activate visnav`

Download data files from my [Google Drive folder](https://drive.google.com/drive/folders/0ByfhOdRO_959X05jTWczWGxLUkk?usp=sharing)
into `data/` folder

Check the file `src/settings.py` and change e.g. the `logs` and `cache` folder paths. The simulation creates a lot of data into those folders.
If you run e.g. a simulation of 1000 iterations, 3GB of log data (images mainly) is created. Around 4GB is used to cache
the generated random situations, including related navcam images and noisy shape models.

To run standalone GUI mode (doesn't work at the moment):<br/>
`python src/visnav.py`

To run a Monte Carlo batch, open batch1.py in an editor to see argument options, then run e.g.:<br/>
`python src/batch1.py didy1w akaze+centroid+smn 10`

Note: There's currently a bug with running the algorithms at the same time while creating the cached data.
The workaround is to:
 * set `ONLY_POPULATE_CACHE = True` at `settings.py`
 * run `src/batch1.py` with the desired arguments to generate cache contents
 * set `ONLY_POPULATE_CACHE = False` at `settings.py`
 * run `src/batch1.py` with the same arguments to run the simulation and generate the log

## Documentation
This work started as a project done at Aalto University, School of Electrical Engineering.
The documentation done towards those credits can be found [here](https://docs.google.com/document/d/1lXqXdR02dAcGPsClwZOXj39RbBfrcscxIKrUyMY_WGU/edit#heading=h.dw2dac9r7xzm).
Sadly, that document might be quite obsolete by now.

I find `hg` easier to use than `git`, so for this repo I've used `hg` with `hggit` extension.
Seems that it didn't change `.hgignore` file into `.gitignore`...
