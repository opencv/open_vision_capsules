## Introduction

The Capsule class provides information to the application about what a capsule
is and how it should be run. Every capsule defines a Capsule class that extends
BaseCapsule in its `capsule.py` file.

```python
from vcap import (
    BaseCapsule,
    NodeDescription,
    options
)

class Capsule(BaseCapsule):
    name = "detector_person"
    version = 1
    stream_state = StreamState
    input_type = NodeDescription(size=NodeDescription.Size.NONE),
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = backend_loader_func
    options = {
        "threshold": options.FloatOption(
            default=0.5,
            min_val=0.1,
            max_val=1.0)
    }
```

## Fields

### name

```python
name: str
```

The name of the capsule. This value uniquely defines the capsule and cannot be
shared by other capsules. By convention, the capsule's name is prefixed by some
short description of the role it plays ("detector", "recognizer", etc) followed
by the kind of data it relates to ("person", "face", etc) and, if necessary,
some differentiating factor ("fast", "close_up", "retail", etc).


### version

```python
version: int
```

The incremental version of the capsule. It's recommended to change the version
on any change of capsule options, algorithms or models.

### device_mapper

```python
device_mapper: DeviceMapper
```

A device mapper contains a single field, `filter_func`, which is a function that
takes in a list of all available device strings and returns a list of device
strings that are compatible with this capsule.

### stream_state

```python
stream_state: Type[BaseStreamState]
```
(Optional) A class that the application will make instances of for each video
stream. See [the section on StreamState](../stream_state/) for more information.

### input_type

```python
input_type: NodeDescription
```

Describes the types of DetectionNodes that this capsule takes in as input. See
[the section on inputs and outputs](../inputs_and_outputs/)
for more information.

### output_type

```python
output_type: NodeDescription
```

Describes the types of DetectionNodes that this capsule produces as output. See
[the section on inputs and outputs](../inputs_and_outputs/)
for more information.

### backend_loader

```python
backend_loader: Callable[[dict, str], BaseBackend]
```

A function that creates a backend for this capsule. Takes the following as
arguments:

- `capsule_files`: Provides access to all files in the capsule. The keys are
  file names and the values are `bytes`.
- `device`: A string specifying the device that this backend should use. For
  example, the first GPU device is specified as "GPU:0".

The function must return a class that subclasses BaseBackend.
See [the section on backends](../backends/) for more information.

### options

```python
options: dict[str, Option]
```

A dict where the key is an option name and the value is an Option object. See
[the section on options](../options/) for more information.
