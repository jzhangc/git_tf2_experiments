"""other utilities"""

# ------ modules ------
import os
import sys
import argparse
from typing import Union, Optional
from itertools import chain


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


class AppArgParser(argparse.ArgumentParser):
    """
    # Purpose
        The help page will display when (1) no argumment was provided, or (2) there is an error
    """

    def error(self, message, *lines):
        string = "\n{}ERROR: " + message + "{}\n" + \
            "\n".join(lines) + ("{}\n" if lines else "{}")
        print(string.format(colr.RED_B, colr.RED, colr.ENDC))
        self.print_help()
        sys.exit(2)


# ------ functions -------
def addBoolArg(parser, name, help, input_type, default=False):
    """
    # Purpose\n
        automatically add a pair of mutually exclusive boolean arguments to the
        argparser
    # Arguments\n
        parser: a parser object.\n
        name: str. the argument name.\n
        help: str. the help message.\n
        input_type: str. the value type for the argument\n
        default: the default value of the argument if not set\n
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name,
                       action='store_true', help=input_type + '. ' + help)
    group.add_argument('--no-' + name, dest=name,
                       action='store_false', help=input_type + '. ''(Not to) ' + help)
    parser.set_defaults(**{name: default})


def csvPath(string):
    # # (inactive) below: relative to the script dir
    # script_path = os.path.dirname(__file__)
    # full_path = os.path.normpath(os.path.join(script_path, string))

    # below: relative to working dir
    # use os.path.expanduser to understand "~"
    full_path = os.path.normpath(os.path.abspath(os.path.expanduser(string)))

    if os.path.isfile(full_path):
        # return full_path
        _, file_ext = os.path.splitext(full_path)
        if file_ext != '.csv':
            raise ValueError('Input file needs to be .csv type.')
        else:
            return full_path
    else:
        raise ValueError('Invalid input file or input file not found.')


def fileDir(string):
    # # (inactive) below: relative to the script dir
    # script_path = os.path.dirname(__file__)
    # full_path = os.path.normpath(os.path.join(script_path, string))

    # below: relative to working dir
    # use os.path.expanduser to understand "~"
    full_path = os.path.normpath(os.path.abspath(os.path.expanduser(string)))

    if os.path.isdir(full_path):
        return full_path
    else:
        raise ValueError('Directory not found.')


def flatten(x): return [item for sublist in x for item in sublist]


def good_update_interval(total_iters, num_desired_updates):
    '''
    This function will try to pick an intelligent progress update interval
    based on the magnitude of the total iterations.

    Parameters\n
      `total_iters` - The number of iterations in the for-loop.
      `num_desired_updates` - How many times we want to see an update over the
                              course of the for-loop.
    '''
    # Divide the total iterations by the desired number of updates. Most likely
    # this will be some ugly number.
    exact_interval = total_iters / num_desired_updates

    # The `round` function has the ability to round down a number to, e.g., the
    # nearest thousandth: round(exact_interval, -3)
    #
    # To determine the magnitude to round to, find the magnitude of the total,
    # and then go one magnitude below that.

    # Get the order of magnitude of the total.
    order_of_mag = len(str(total_iters)) - 1

    # Our update interval should be rounded to an order of magnitude smaller.
    round_mag = order_of_mag - 1

    # Round down and cast to an int.
    update_interval = int(round(exact_interval, -round_mag))

    # Don't allow the interval to be zero!
    if update_interval == 0:
        update_interval = 1

    return update_interval


def string_flex(s: str, idx: Union[int, str, list]):
    """
    # Purpose\n
        String slicing with flexible index styles.\n

    # Arguments\n
        s: str. Input string.\n
        idx: int, str, list. Flexbile style indices.\n

    # Details\n
        `idx` can take many forms.
            - single int, e.g. `2`, `-3`
            - list of int, e.g. `[2, 3, 6]`
            - single string, e.g. `'2'`, `'2:6'`, `"2:"`, `":2"`, `"-1"`, `"label1"`
            - list of string, e.g. `["2", "3", "4"]`, `["label1", "label2". "label3"]`
            - Note, the following are NOT SUPPORTED: `["1, 2, 3"]`, or  `"label1label2lable3"` (three labels, will be treated as one label: `label1label2lable3`). 
            However, int strings like `"123"` are supported, e.g. `"123"` as three ints. 
    """
    try:
        o = eval(f's[{idx}]')
    except Exception as e:
        real_idx = []
        try:
            for i in idx:
                real_idx.append(s.index(i))
        except ValueError as e1:
            for i in idx:
                real_idx.append(i)
                real_idx = [int(x) for x in real_idx]
        except Exception as e2:
            real_idx.append(s.index(idx))
        o = [s[x] for x in real_idx]

    return o


def zip_equal(*iterables):
    """"stole from: https://stackoverflow.com/questions/32954486/zip-iterators-asserting-for-equal-length-in-python"""

    # For trivial cases, use pure zip.
    if len(iterables) < 2:
        return zip(*iterables)

    # Tail for the first iterable
    first_stopped = False

    def first_tail():
        nonlocal first_stopped
        first_stopped = True
        yield
        return

    # Tail for the zip
    def zip_tail():
        if not first_stopped:
            raise ValueError('zip_equal: first iterable is longer')
        for _ in chain.from_iterable(rest):
            yield
            raise ValueError('zip_equal: first iterable is shorter')

    # Put the pieces together
    iterables = iter(iterables)
    first = chain(next(iterables), first_tail())
    rest = list(map(iter, iterables))
    return chain(zip(first, *rest), zip_tail())
