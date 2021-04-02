# based on io_utils.py from tensorflow (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/io_utils.py)

import numpy as np
import six

def ask_to_proceed_with_overwrite(filepath):
    """Produces a prompt asking about overwriting a file.
    Arguments:
      filepath: the path to the file to be overwritten.
    Returns:
      True if we can proceed with overwrite, False otherwise.
    """
    overwrite = six.moves.input('[WARNING] %s already exists - overwrite? '
                              '[y/n]' % (filepath)).strip().lower()
    while overwrite not in ('y', 'n'):
        overwrite = six.moves.input('Enter "y" (overwrite) or "n" '
                                '(cancel).').strip().lower()
    if overwrite == 'n':
        return False

    print('[TIP] Next time specify overwrite=True!')
    return True