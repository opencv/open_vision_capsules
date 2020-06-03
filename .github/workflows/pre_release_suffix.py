"""Prints a suffix that will be appended to the end of the version strings
for the vcap and vcap-utils packages.

For example, if there have been 3 commits since the last tagged release and the
current version string in the setup.py is '0.1.3', the suffix will be '.dev3'.
The resulting package version will then become '0.1.3.dev3'.

For commits that have a tag, this script will print nothing. That way, the
resulting package will have no pre-release suffix.

Ideally, we would be using setuptools_scm to manage all of this for us.
Unfortunately, setuptools_scm doesn't work for repositories with multiple
packages. See: https://github.com/pypa/setuptools_scm/issues/357
"""

import subprocess
import re
import sys

TAG_PATTERN = r"^v\d+\.\d+\.+\d+-?(\d+)?-?(.+)?$"

result = subprocess.run(["git", "describe", "--tags"],
                        check=True, stdout=subprocess.PIPE, encoding="utf-8")

match = re.match(TAG_PATTERN, result.stdout)
if match is None:
    print(f"Could not match tag: '{result.stdout}'", file=sys.stderr)
    sys.exit(1)

if match.group(1) is not None:
    print(f".dev{match.group(1)}")
else:
    print("")
