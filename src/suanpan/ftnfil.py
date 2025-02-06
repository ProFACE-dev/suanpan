# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

"""low level access to .fil files"""

import array
import mmap
import os

import numpy as np

ARECL = 4096  # fixed size of Abaqus records
AWL = 8  # Abaqus word length
AWR = ARECL // AWL  # Abaqus words per record

# dtype used for low level access to binary '.fil'.
# The file is organized in blocks:
# pad data (4 bytes), 512 x 8 bytes "words", pad data (4 bytes)
# pad data is ARECL stored as "u4"
# data is stored as "512 V8"
ABQ = np.dtype([("pad0", "=u4"), ("data", f"V{AWL}", AWR), ("pad1", "=u4")])
assert ABQ.itemsize == ARECL + 8


def mmfil(path):
    """factory function that creates a mmapped view of '.fil' file

    input arguments:
    path -- path-like object pointing to the abaqus '.fil' file

    output:
    read-only structured numpy ndarray with a mmapped view of input file
    with ABQ dtype
    """

    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    os.close(fd)

    if len(mm) % ABQ.itemsize:
        msg = f"invalid file length, should be a multiple of {ABQ.itemsize:d}"
        raise ValueError(msg)

    arr = np.ndarray(shape=(len(mm) // ABQ.itemsize,), dtype=ABQ, buffer=mm)

    # minimal sanity check
    # check if first and last 10 records have correct pad data
    if (arr["pad0"][:10] != ARECL).any() or (arr["pad1"][:10] != ARECL).any():
        msg = "invalid file format: wrong pad data"
        raise ValueError(msg)

    return arr


def rstream(data, pos=0):
    if not isinstance(data, np.ndarray) or data.dtype != "V8":
        msg = "'data' is not an array with 'V8' dtype"
        raise TypeError(msg)

    df = data.flat
    uf = data.view("2u4")[..., 0].flat
    assert len(uf) == len(df)

    end = len(uf)
    skip = None
    try:
        rlen, rtyp = uf[pos : pos + 2]
        while True:
            data = df[pos + 2 : pos + rlen]
            skip = yield pos, rtyp, rlen, data
            pos += rlen
            if skip is not None:
                if skip == ():
                    _rlen, _rtyp = rlen, rtyp
                    while True:
                        rlen, rtyp = uf[pos : pos + 2]
                        if (rlen, rtyp) != (_rlen, _rtyp):
                            break
                        pos += rlen
                else:
                    # skip to first record with 'rtyp in skip'
                    while True:
                        rlen, rtyp = uf[pos : pos + 2]
                        if rtyp in skip:
                            break
                        pos += rlen
            else:
                rlen, rtyp = uf[pos : pos + 2]
    except ValueError:
        assert pos == end
        pass


def walkr(data, start=0):
    """return iterator on abaqus records"""

    if not isinstance(data, np.ndarray) or data.dtype != "V8":
        msg = "'data' is not an array with 'V8' dtype"
        raise TypeError(msg)

    df = data.flat
    uf = data.view("2u4")[..., 0].flat
    assert len(uf) == len(df)

    pos = start
    end = len(uf)
    while pos < end:
        rlen = uf[pos]
        req = yield pos, (uf[pos + 1], rlen - 2)
        if req:
            yield df[pos + 2 : pos + rlen]
        pos += rlen
    assert pos == end


def datablock(data, start, end, rlen):
    """return the data portion data in a record block

    input args:
    start, end: logical boundaries of record block
    rlen: record length

    output:
    array of records of type V((rlen-2)*AWL)
    """

    if not isinstance(data, np.ndarray) or data.dtype != "V8":
        msg = "'data' is not an array with 'V8' dtype"
        raise TypeError(msg)

    return (
        data.flat[start:end]
        .reshape(-1, rlen)[..., 2:]
        .ravel()
        .view(f"V{(rlen-2)*AWL}")
    )


def makeidx(data):
    """make an index to the position of contigous records of same type"""

    if not isinstance(data, np.ndarray) or data.dtype != "V8":
        msg = "'data' is not an array with 'V8' dtype"
        raise TypeError(msg)

    idx = []
    stream = rstream(data)
    pos, key, *_ = next(stream)
    while True:
        idx.append((key, pos))
        try:
            pos, key, *_ = stream.send(())
        except StopIteration:
            break
    return idx


def incstart(data, rec):
    """find increment start (record type 2000)
    in 'data' starting from fortran record number 'rec'
    """

    RLEN = 23
    RTYP = 2000

    if not isinstance(data, np.ndarray) or data.dtype != "V8":
        msg = "'data' is not an array with 'V8' dtype"
        raise TypeError(msg)

    data_uv = data.view("2u4")[..., 0]
    assert data_uv.shape == data.shape
    assert data_uv.shape[1] == AWR

    step_rec = array.array("L")
    step_data = bytearray()
    for i in range(rec, len(data)):
        if tuple(data_uv[i][:2]) == (RLEN, RTYP):
            step_rec.append(i)
            step_data.extend(data[i][2:RLEN])
    # add last item, so that 'start, end = step_rec[i:i+2]'
    step_rec.append(len(data))
    return step_rec, step_data
