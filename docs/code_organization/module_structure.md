# Module structure in src/
## main.py
The entrypoint of the program is the `main.py` file. It should be called via terminal with `python main.py` or similar. The program is controlled via UNIX flags. Just try to run the program and it will guide you.

## src/data/
All dataset images should go under the `data` folder. Concrete structure is to be set.

## src/models/
A variety of models will be defined here. They should be imported with `from models import xxx`. This is just the class definitions, though. The weights should be stored in the `checkpoints/` directory under the root of the project.

## src/utils/
Helper modules go here.
