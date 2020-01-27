## Introduction

Capsules are defined by the data they take as input and the information they
give as output. Applications use this information to connect capsules to each
other and schedule their execution. These inputs and outputs are _defined_
by NodeDescription objects and _realized_ by DetectionNode objects.

## DetectionNode

Capsules use DetectionNode objects to communicate results to other capsules and
the application itself. A DetectionNode contains information on a detection in
the current frame. Capsules that detect objects in a frame create new
DetectionNodes. Capsules that discover attributes about detections add data to
existing DetectionNodes.

A DetectionNode object contains the following fields:

### class_name

```python
class_name: str
```

The detection class name. This describes what the detection is. A detection of
a person would have a name="person".

### coords

```python
coords: List[List[int]]
```

A list of coordinates defining the detection as a polygon in-frame. Comes in
the format `[[x,y], [x,y]...]`.


### attributes

```python
attributes: Dict[str, str]
```

A key-value store where the key is the type of attribute being described and
the value is the attribute's value. For instance, a capsule that detects gender
might add a "gender" key to this dict, with a value of either "masculine" or
"feminine".

### encoding

```python
encoding: Optional[numpy.ndarray]
```

An array of float values that represent an encoding of the detection. This can
be used to recognize specific instances of a class. For instance, given a
picture of person’s face, the encoding of that face and the encodings of future
faces can be compared to find that person in the future.

### track_id

```python
track_id: Optional[UUID]
```

If this object is tracked, this is the unique identifier for this detection
node that ties it to other detection nodes in future and past frames (within
the same stream).

### extra_data

```python
extra_data: Dict[str, object]
```

A dict of miscellaneous data. This data is provided directly to clients without
modification, so it’s a good way to pass extra information from a capsule to
other applications.

### children

```python
children: List[DetectionNode]
```

A list of DetectionNode objects that are child detections of this detection.
For example, a face DetectionNode might be a child of a person DetectionNode.

## NodeDescription

Capsules use NodeDescriptions to describe the kinds of DetectionNodes they
take in as input and produce as output.

A capsule may take a DetectionNode as input and produce zero or more
DetectionNodes as output. Capsules define what information inputted
DetectionNodes must have and what information outputted detection nodes will
have using NodeDescriptions.

A NodeDescription has the following fields:

### size

```python
size: NodeDescription.Size
```

Specifies how many DetectionNodes the capsule takes as input at once.

- `NodeDescription.Size.NONE`: The capsule does not take any input. This is
  common for capsules that detect objects in frame. These algorithms usually
  only need the video frame.
- `NodeDescription.Size.SINGLE`: The capsule takes a single DetectionNode
  object. This is common for capsules that find attributes for objects that have
  been detected by other capsules.
- `NodeDescription.Size.ALL`: The capsule takes all available DetectionNodes
  that fit the capsule's input requirements. This is common for capsules that
  track objects between video frames.

### detections

```python
detections: List[str]
```

A list of detection class names. This field is used to describe a
DetectionNodes that have been detected as one of these class names.

For example, a capsule that can encode cars or trucks would use a
NodeDescription like this as its `input_type`:

```python
NodeDescription(detections=["car", "truck"])
```

A capsule that can detect people and dogs would use a NodeDescription like this
as its `output_type`:

```python
NodeDescription(detections=["person", "dog"])
```

### attributes

```python
attributes: Dict[str, List[str]]
```

A dict whose key is an attribute name and whose value is all possible values
for that attribute. This field is used to describe DetectionNodes that has a
value for every specified attribute.

For example, a capsule that operates on detections that have been classified for
gender use a NodeDescription like this as its `input_type`:

```python
NodeDescription(
    attributes={
        "gender": ["male", "female"],
        "color": ["red", "blue", "green"]
    })
```

A capsule that can classify people’s gender as either male or female would have
the following NodeDescription as its `output_type`:

```python
NodeDescription(
    detections=["person"],
    attributes={
        "gender": ["male", "female"]
    })
```

### encoded

```python
encoded: bool
```

True if a DetectionNode described by this NodeDescription is encoded.

For example, a capsule that operates on detections of cars that have been
encoded use a NodeDescription like this as its `input_type`:

```python
NodeDescription(
    detections=["car"],
    encoded=True)
```

A capsule that encodes people would use a NodeDescription like this as its
`output_type`:

```python
NodeDescription(
    detections=["person"],
    encoded=True)
```

### tracked

True if a DetectionNode described by this NodeDescription is encoded.

For example, a capsule that operates on person detections that have been tracked
would use a NodeDescription like this as its `input_type`.

```python
NodeDescription(
    detections=["person"],
    tracked=True)
```

A capsule that tracks people would use a NodeDescription like this as its
`output_type`:

```python
NodeDescription(
    detections=["person"],
    tracked=True)
```

### extra_data

```python
extra_data: List[str]
```

A list of keys in a DetectionNode’s `extra_data` field. This field is used to
describe DetectionNodes that have a value for every specified `extra_data` key.

For example, a capsule that operates on people detections with a
"process_extra_fast" `extra_data` field would use a NodeDescription like this
as its `input_type`:

```python
NodeDescription(
    detections=["person"],
    extra_data=["process_extra_fast"])
```

A capsule that adds an "is_special" `extra_data` field to its person-detected
output would use a NodeDescription like this as its `output_type`:

```python
NodeDescription(
    detections=["person"],
    extra_data=["is_special"])
```
