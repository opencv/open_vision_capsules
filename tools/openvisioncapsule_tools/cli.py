import os
os.environ['BF_LOG_PRINT'] = 'TRUE'

import signal
import sys
from argparse import ArgumentParser

import i18n
import yaml
from openvisioncapsule_tools import print_utils, command_utils
from openvisioncapsule_tools.capsule_packaging import capsule_packaging
from openvisioncapsule_tools.capsule_infer import capsule_infer


def add_translations(translations, prefix=""):
    for key, value in translations.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            add_translations(value, full_key)
        else:
            i18n.add_translation(full_key, value)


def cli_main():
    i18n.load_path.append(_TRANSLATIONS_PATH)

    # Load translations manually
    translation_file = os.path.join(_TRANSLATIONS_PATH, "portal.en.yml")
    if os.path.exists(translation_file):
        with open(translation_file, "r") as file:
            translations = yaml.safe_load(file)
            add_translations(translations)

    parser = ArgumentParser(
        description=i18n.t("en.portal.description"), usage=i18n.t("en.portal.usage")
    )

    parser.add_argument(
        "command", default=None, nargs="?", help=i18n.t("en.portal.command-help")
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
