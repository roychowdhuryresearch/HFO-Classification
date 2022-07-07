
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

def arg_parse(args):
    # setting the log level on the root logger must happen BEFORE any output

    # parse values from a configuration file if provided and use those as the
    # default values for the argparse arguments
    config_argparse = ArgumentParser(prog=__file__, add_help=False)
    config_argparse.add_argument('-c', '--config-file',
                                 help='path to configuration file', required=True)
    config_args, _ = config_argparse.parse_known_args(args)

    defaults = {}

    if config_args.config_file:
        try:
            config_parser = ConfigParser()
            with open(config_args.config_file) as f:
                config_parser.read_file(f)
            config_parser.read(config_args.config_file)
        except OSError as err:
            sys.exit(1)
        defaults.update(dict(config_parser.items('options')))

    # parse the program's main arguments using the dictionary of defaults and
    # the previous parsers as "parent' parsers
    parsers = [config_argparse]
    main_parser = ArgumentParser(prog=__file__, parents=parsers)
    main_parser.set_defaults(**defaults)
    main_parser.add_argument('-d10', '--datain_10')
    main_parser.add_argument('-o10', '--dataout_10')
    main_parser.add_argument('-d90', '--datain_90')
    main_parser.add_argument('-o90', '--dataout_90')
    main_parser.add_argument('-cha', '--channel_annotation')
    main_parser.add_argument('-i90', '--inference_res')
    main_args = main_parser.parse_args(args)
    return main_args

if __name__ == "__main__":
    arg_parse(sys.argv[1:])