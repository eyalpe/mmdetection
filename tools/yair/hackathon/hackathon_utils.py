import os


def get_mmdet_root():
    """Find the mmdetection code root directory.

       Returns:
           (str): Directory path of the mmdetection code root directory.
       """
    return os.path.abspath(os.path.dirname(__file__)+'/../../../')
