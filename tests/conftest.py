# Multiprocessing is used inside RAMLoadedDataset.
# pytest uses threads, which result in a DeprecationWarning
# when testing RAMLoadedDataset. The safe (albeit much slower)
# alternative is using "forkserver" as a mp.context
# only in the test suite, since the program can actually use
# "fork" without problems.
import multiprocessing as mp
mp.set_start_method("forkserver")
