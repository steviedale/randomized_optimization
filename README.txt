Instructions on how to run:

Clone repo:
`git clone --recurse-submodules -j8 https://github.com/steviedale/randomized_optimization.git`

Change Dir into repo.

Create virtual environment:
`python3 -m venv venv`

Activate environment:
`source venv/bin/activate`

Pip install reps
`pip install -r requirements.txt`

For each folder in ['four_peaks', 'queens', 'nn']:

	- Each `.py` file runs a specific experiment. Hyperparameters are already set, but you can change them if you want to. Run python file.
	- Each `plot_...ipynb` file creates the plots for each experiment. Run these scripts to generate images and tables.