import os
os.environ['BF_LOG_PRINT'] = 'TRUE'

import signal
import sys
from argparse import ArgumentParser

import i18n
from openvisioncapsule_tools import print_utils, command_utils
from openvisioncapsule_tools.capsule_packaging import capsule_packaging


def cli_main():
    i18n.load_path.append(_TRANSLATIONS_PATH)

    parser = ArgumentParser(
        description=i18n.t("portal.description"), usage=i18n.t("portal.usage")
    )

    parser.add_argument(
        "command", default=None, nargs="?", help=i18n.t("portal.command-help")
    )

    args = parser.parse_args(sys.argv[1:2])

    if args.command is None:
        parser.print_help()
    elif args.command in command_utils.by_name:
        command = command_utils.by_name[args.command]
        command()
    else:
        error_message = i18n.t("portal.unknown-command")
        error_message = error_message.format(command=args.command)
        print_utils.print_color(
            error_message, color=print_utils.Color.RED, file=sys.stderr
        )
        parser.print_help()


_TRANSLATIONS_PATH = os.path.join(os.path.dirname(__file__), "translations")

if __name__ == "__main__":
    cli_main()
