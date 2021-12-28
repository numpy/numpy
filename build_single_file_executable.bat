call activate numpy-dev39
python -m pip wheel -w . .
pipenv install pyoxidizer
pipenv run pyoxidizer run installer
