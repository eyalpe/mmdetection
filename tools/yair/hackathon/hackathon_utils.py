import os


def get_mmdet_root():
    """Find the mmdetection code root directory.

       Returns:
           (str): Directory path of the mmdetection code root directory.
       """
    return os.path.abspath(os.path.dirname(__file__)+'/../../../')


def get_user_home_dir():
    """Find the current user's home directory.

       Returns:
           (str): Current user's home directory path.
       """
    return os.path.expanduser('~')
