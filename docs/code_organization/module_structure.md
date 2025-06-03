# Module structure in src/
## main.py
The entrypoint of the program is the `main.py` file directly under `src`. During development, I use [uv](https://docs.astral.sh/uv/) to run it: `uv run u2fold [flags]`.

The program is controlled via UNIX-like flags. Just try to run it with the flag `--help` and it will guide you.

## u2fold
The folder `src/u2fold` holds most of the library code for the program. It is split into several Python subpackages (folders) containing different functionalities. 

### CLI
The command line interface (CLI) is implemented in the `cli_parsing` subpackage. Inside it, arguments for the program that are not specific to any particular neural network (log directory, weight directory...) are defined. This model also makes use of the [tag](#class-tracking) utility for registering CLI arguments. The defined CLI arguments are each implemented in their own, separate class, and provide their own validation logic.

The subpackage also exposes the `build_parser` API, which queries tags from different parts of the program to automatically add them to an `argparse.ArgumentParser` object (this is from Python's standard library, cf. [the docs](https://docs.python.org/3/library/argparse.html)). This object encapsulates help messages, argument types and all of the CLI input logic. If a positionally valid set of arguments is provided, it is parsed into a Python `Namespace`.

### Configuration validation and parsing
The logic for validating input CLI arguments and instantiating the respective configuration classes is encapsulated in the `config_parsing` subpackage.

In this context, a *configuration class* is a class that contains information which determines the behaviour of the program at an algorithmic level. That is, whether the program will train a set of models or just execute an already trained one, or which directories the program should access for weights, logs, etc. These are usually implemented via Python's dataclasses (also part of the standard library, cf. [the docs](https://docs.python.org/3/library/dataclasses.html)).

This also includes the hyperparameters used for the models. In fact, each model in U2Fold must provide a configuration class which inherits from `ModelConfig` (more on this in the [models section](#models)).

In this module, configuration classes for both training and execution modes are defined. They both need to be instantiated with an instance of `ModelConfig`.

It exposes the `parse_and_validate_config` API, which takes the resulting `Namespace` from the [CLI](#cli) module, validates the information in it (this means raising an exception with an appropriate error message if some of the provided values makes no sense in the context of the program; for instance, a dropout of 2 or -50 training epochs) and instantiate the appropriate configuration classes.

The implementation of this API is mostly automated, in the sense that there is no manual enumeration of the arguments involved besides a few exceptions which have custom logic that was hard to fit elsewhere. This is achieved via [tag querying](#class-tracking) and leveraging `dataclasses`' `fields` API, as well as the `post_init` dunder method for validation.

### Models
Inside the `models` subpackage lies all the logic regarding strictly neural networks and instantiation. This does **not** include logic for training them nor the minimization loop, as that requires a much more involved logic that is not the responsibility of the neural network class. This was a very deliberate design choice, and that logic is instead implemented in the [orchestrator](#orchestration).

Each neural network in this program is implemented by subclassing the `Model` class, whose implementation is inside the very well-documented module [`generic`](/src/models/generic.py) (see the docstrings there for details and usage examples). Each `Model` subclass must also provide its own `ModelConfig` subclass containing all of the hyperparameters needed to instantiate the model, and the constructor of the subclassed `Model` must take as parameters only an instance of its associated `ModelConfig` and the `device` in which to load the model.

Note: passing `device` as a parameter is a requirement for being able to instantiate the model with pre-existing weights without having them randomly initialized and then instantly replaced by the pre-existing weights; that bugged me out a lot. See [this pytorch tutorial](https://docs.pytorch.org/tutorials/prototype/skip_param_init.html) for details.

### Orchestration
WIP.

### Utilities
In the `utils` subpackage there is a variety of functions used throughout the code that for some reason or another did not fit elsewhere.

#### Class tracking
This particular utility is ubiquitous in this program. It is implemented in the `track.py` submodule. The API it exposes is composed of the `tag`, `get_tag_group` and `get_from_tag` methods.

The `tag` method is a class decorator that, when used, stores one or more references to the decorated class via nested `dict`s. The keys of these nested dictionaries are determined via the tags provided when calling the decorator as positional arguments.

Each of these tags is composed by one or more words separated by slashes ("/").
Each of these words corresponds to a level of nesting; the rightmost one being the key of the final dictionary, whose associated value is the tagged class. For instance, the tag

```python
>>> @tag("my_group/my_class")
>>> class MyClass: ...
```

produces the following nested dictionary:

```python
{"my_group": {"my_class": MyClass}}
```

This decorator is very useful for grouping classes and querying them somewhere else in the program, via for instance `get_tag_group`:

```python
>>> @tag("my_group/my_other_class")
>>> class MyOtherClass: ...
>>> for name, class_ in get_tag_group("my_group").items():
...     print(f"{name} -> {class_}")
...
my_class -> <class '__main__.MyClass'>
my_other_class -> <class '__main__.MyOtherClass'>
```

Finally, `get_from_tag` allows one to directly look up a class. With the previous examples:

```python
>>> get_from_tag("my_group/my_class") is MyClass
True
```

#### Other utilities
The other utilities are not nearly as important and varied in nature. A few examples:

- `singleton_metaclasses` contains two metaclasses that allow the imposition of the singleton pattern onto constructed classes, even if they are abstract. Class inheriting from an `AbstractSingleton` will automatically follow the singleton pattern without manually specifying it.
- `sliding_window` contains a method for comfortably creating an iterable that yields, in order, the 2-grams of a given input iterator. For instance: `["foo", "bar", "baz"] -> (("foo", "bar"), ("bar", "baz"))`. Inspired by Rust's standard library [windows](https://doc.rust-lang.org/std/slice/struct.Windows.html) method on slices.
- `name_conversions` contains utility methods for converting between snake case and "CLI case" (kebab case with two prepended scores).
