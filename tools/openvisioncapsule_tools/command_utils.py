import sys
from argparse import ArgumentParser

by_name = {}
"""A dict that maps command names to their corresponding function"""


def command(name):
    """A decorator that associates command functions to their command name."""

    def wrapper(function):
        by_name[name] = function

    return wrapper


def subcommand_parse_args(parser: ArgumentParser):
    arg_list = sys.argv[2:]
    args = parser.parse_args(arg_list)

    # Run in non-interactive mode if any flags were provided
    if len(arg_list) > 0:
        args.noninteractive = True

    return args
