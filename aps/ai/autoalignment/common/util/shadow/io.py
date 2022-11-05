import os
import contextlib

@contextlib.contextmanager
def redirected_output(stdout_fname='stdout.txt', stderr_fname='stderr.txt'):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with redirected_output():
        out_beam = focusing_system.get_photon_beam()

    Pass stdout_fname=os.devnull and/or stderr_fname=os.devnull if saving to a file is not desired.
    """

    try:
        # bitwise OR
        create_or_write = os.O_CREAT | os.O_RDWR
        file_descriptors = [os.open(f, create_or_write)
                            for f in [stdout_fname, stderr_fname]]
        old_descriptors = os.dup(1), os.dup(2)

        os.dup2(file_descriptors[0], 1)
        os.dup2(file_descriptors[1], 2)

        yield
    finally:
        os.dup2(old_descriptors[0], 1)
        os.dup2(old_descriptors[1], 2)
        # close the temporary fds
        os.close(file_descriptors[0])
        os.close(file_descriptors[1])