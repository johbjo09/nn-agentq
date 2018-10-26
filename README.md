# NN Snake trainer and client
#

To train faster, use the C++ client to run games.

## Running the client
You need to have [`pipenv`](pipenv) installed as that is how we manage both
dependencies and virtualenvs for the project. To install `pipenv`:
```
sudo pip install pipenv
```

Once you have `piipenv` installed you can use it to download all dependencies
and activate the virtualenv:
```
pipenv shell
```

This will put you in a shell with the python environment activated, allowing you
to run the client:
```
python agentQ
```

