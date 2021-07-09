"""other utilities"""

# ------ modules ------
import os
import sys


# ------ classes -------
class colr:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'  # end colour


# ------ functions -------
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


def addBoolArg(parser, name, help, input_type, default=False):
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


def csvPath(string):
    # # below: relative to the script dir
    # script_path = os.path.dirname(__file__)
    # full_path = os.path.normpath(os.path.join(script_path, string))

    # below: relative to the working dir
    # use os.path.expanduser to understand "~"
    full_path = os.path.normpath(os.path.abspath(os.path.expanduser(string)))

    if os.path.isfile(full_path):
        # return full_path
        _, file_ext = os.path.splitext(full_path)
        if file_ext != '.csv':
            error('Input file needs to be .csv type.')
        else:
            return full_path
    else:
        error('Invalid input file or input file not found.')


def fileDir(string):
    # # below: relative to the script dir
    # script_path = os.path.dirname(__file__)
    # full_path = os.path.normpath(os.path.join(script_path, string))

    # below: relative to the working dir
    # use os.path.expanduser to understand "~"
    full_path = os.path.normpath(os.path.abspath(os.path.expanduser(string)))

    if os.path.isdir(full_path):
        return full_path
    else:
        error("Directory not found.")
