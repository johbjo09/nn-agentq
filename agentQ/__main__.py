import argparse
import logging

import colorlog

from genetic_controller import GeneticController

log_levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'fatal': logging.FATAL
}
log_names = list(log_levels)


def set_up_logging(args):
    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        fmt=('%(log_color)s[%(asctime)s %(levelname)8s] --'
             ' %(message)s (%(filename)s:%(lineno)s)'),
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    if args.log_level in log_levels:
        logging.basicConfig(
            handlers=[handler],
            level=log_levels[args.log_level]
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Python client for Cygni's snakebot competition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Server flags
    parser.add_argument(
        '-r',
        '--host',
        default='localhost',
        help='The host to connect to')
    parser.add_argument(
        '-p', '--port', default='8080', help='The port to connect to')
    parser.add_argument(
        '-v',
        '--venue',
        default='training',
        choices=['training', 'tournament'],
        help='The venue (training or tournament)')

    # Verbosity
    parser.add_argument(
        '-l',
        '--log-level',
        default=logging.INFO,
        choices=log_names,
        help='The log level for the client')

    return parser.parse_args()


def main():
    args = parse_args()
    set_up_logging(args)

    genetic_controller = GeneticController(args)

    genetic_controller.run_simulation()


if __name__ == "__main__":
    main()
