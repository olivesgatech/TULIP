# TULIP
Welcome to the TULIP repository! To start experimenting, create your own virtual environment
and install the dependencies with poetry. You can do this by running

```
pip install -U pip
pip install poetry
poetry install
```

This should install most dependencies you will need. Since this code uses absolute paths, you 
will need to export the PYTHONPATH to this repository before running any experiment.

```
export PYTHONPATH=$PYTHONPATH:$PWD
```

This command assumes you already moved into this repository with cd.

To run an experiment, you must specify your configurations in the config file. We provide a
simple example_config.toml to show you a few general options you have available. The command
to run an experiment is

```
python3 training/classification/run.py --config <PATH_TO_CONFIG>
```

where PATH_TO_CONFIG is the toml file mentioned earlier. Happy experimenting!
