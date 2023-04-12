from openvisioncapsule_tools.command_utils import command, subcommand_parse_args, by_name
from .capsule_infer import capsule_infer_add_args, capsule_infer

@command("capsule_inference")
def capsule_inference_main():
    parser = capsule_infer_add_args()
    args = subcommand_parse_args(parser)
    capsule_infer(args)


if __name__ == "__main__":
    by_name["capsule_inference"]()

