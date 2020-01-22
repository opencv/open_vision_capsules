## Introduction

Capsules can provide runtime configuration options that change the way the
capsule operates. These options will appear on the client and can also be
changed in the UI. Options have a type and constraints that define what values
are valid.

## FloatOption

An option whose value is a floating point.

| Field Name  | Type  | Description                                          |
|-------------|-------|------------------------------------------------------|
| description | str   | The description for this option                      |
| default     | float | The default value of this option                     |
| min_val     | float | The minimum allowed value for this option, inclusive |
| max_val     | float | The maximum allowed value for this option, inclusive |

## IntOption

An option whose value is an integer.

| Field Name  | Type | Description                                          |
|-------------|------|------------------------------------------------------|
| description | str  | The description for this option                      |
| default     | int  | The default value of this option                     |
| min_val     | int  | The minimum allowed value for this option, inclusive |
| max_val     | int  | The maximum allowed value for this option, inclusive |

## EnumOption

An option with a discrete set of possible string values. This kind of option
would be visualized in a UI as a dropdown.

| Field Name  | Type      | Description                                |
|-------------|-----------|--------------------------------------------|
| description | str       | The description for this option            |
| default     | str       | The default value of this option           |
| choices     | List[str] | A list of all valid values for this option |

## BoolOption

An option whose value is a boolean.

| Field Name  | Type | Description                      |
|-------------|------|----------------------------------|
| description | str  | The description for this option  |
| default     | bool | The default value of this option |
