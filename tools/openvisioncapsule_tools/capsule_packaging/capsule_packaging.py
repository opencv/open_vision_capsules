from vcap.loading.vcap_packaging import packaging, packaging_parse_args
from openvisioncapsule_tools.command_utils import command, subcommand_parse_args, by_name


@command("capsule_packaging")
def capsule_packaging_main():
    parser = packaging_parse_args()
    args = subcommand_parse_args(parser)
    rtn = packaging(args.capsule_dir, args.capsule_file, args.capsule_key)
    if rtn == False:
        parser.print_help()


if __name__ == "__main__":
    by_name["capsule_packaging"]()

