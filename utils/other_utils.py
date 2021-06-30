"""data utilities"""

# ------ modules ------
import os
import sys


# ------ function -------
# below: a lambda funciton to flatten the nested list into a single list
def flatten(x): return [item for sublist in x for item in sublist]


def error(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}ERROR: " + message + "{}\n" + \
        "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(colr.RED_B, colr.RED, colr.ENDC))
    sys.exit(2)


def warn(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(colr.YELLOW_B, colr.YELLOW, colr.ENDC))


def add_bool_arg(parser, name, help, input_type, default=False):
    """
    Purpose\n
                    autmatically add a pair of mutually exclusive boolean arguments to the
                    argparser
    Arguments\n
                    parser: a parser object
                    name: str. the argument name
                    help: str. the help message
                    input_type: str. the value type for the argument
                    default: the default value of the argument if not set
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name,
                       action='store_true', help=input_type + '. ' + help)
    group.add_argument('--no-' + name, dest=name,
                       action='store_false', help=input_type + '. ''(Not to) ' + help)
    parser.set_defaults(**{name: default})


def csv_path(string):
    input_path = os.path.dirname(__file__)
    full_path = os.path.normpath(os.path.join(input_path, string))

    if os.path.isfile(full_path):
        # return full_path
        _, file_ext = os.path.splitext(full_path)
        if file_ext != '.csv':
            error('input file needs to be .csv type')
        else:
            return full_path
    else:
        error('invalid input file or input file not found.')


def output_dir(string):
    input_path = os.path.dirname(__file__)
    full_path = os.path.normpath(os.path.join(input_path, string))

    if os.path.isdir(full_path):
        return full_path
    else:
        error("output directory not found.")
