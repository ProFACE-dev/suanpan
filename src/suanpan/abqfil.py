# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

"""high level interface"""

import array
import itertools
import logging
from collections import namedtuple

import numpy as np

from . import ftnfil

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

# named tuple for 1911 + datarecords
StepDataBlock = namedtuple("StepDataBlock", ("flag", "set", "eltype", "data"))


def _issorted(v):
    """returns true if vector v is sorted"""
    return (v[:-1] <= v[1:]).all()


def _isunique(v):
    """returns true if vector v is sorted with no repetitions"""
    return (v[:-1] < v[1:]).all()


def _pad(x):
    """return min multiple of AWL grater than or equal to x"""
    return x + (-x % ftnfil.AWL)


def _abq_dtype(*items):
    """make abaqus dtype"""

    if not items:
        msg = "The number of items should be grater than zero"
        raise ValueError(msg)

    names, formats = zip(*items)
    formats = tuple(map(np.dtype, formats))
    cumsum = tuple(itertools.accumulate(_pad(i.itemsize) for i in formats))
    assert len(names) == len(formats) == len(cumsum)
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": (0,) + cumsum[:-1],
            "itemsize": cumsum[-1],
        }
    )


ABQR = {
    1911: _abq_dtype(
        ("out_type", "i4"),
        ("out_set", "S8"),
        ("out_element", "S8"),
    ),
    1921: _abq_dtype(
        ("ver", "S8"),
        ("date", "S16"),
        ("time", "S8"),
        ("nelm", "u4"),
        ("nnod", "u4"),
        ("elsiz", "f8"),
    ),
    2000: _abq_dtype(
        ("ttime", "f8"),
        ("stime", "f8"),
        ("cratio", "f8"),
        ("sampl", "f8"),
        ("procid", "i4"),
        ("step", "u4"),
        ("incr", "u4"),
        ("lpert", "i4"),
        ("lpf", "f8"),
        ("freq", "f8"),
        ("tinc", "f8"),
        ("subheading", "S80"),
    ),
}


def _vlenr(rtyp, rlen):
    # first 2 records are not data
    rlen -= 2
    if rtyp == 1501:
        assert rlen >= 5
        items = [
            ("name", "S8"),
            ("sdim", "i4"),
            ("stype", "i4"),
            ("nfacet", "i4"),
            ("nmaster", "i4"),
        ]
        if rlen > 5:
            ## FIXME: Shape-1 fields in dtypes
            items.append(
                ("msurf", f"({rlen - 5:d},)S8" if rlen - 5 > 1 else "S8")
            )
        return _abq_dtype(*items)
    elif rtyp == 1900:
        return np.dtype(
            [("elnum", "i8"), ("eltyp", "S8"), ("ninc", "i8", (rlen - 2,))]
        )
    elif rtyp == 1901:
        return np.dtype([("nnum", "i8"), ("coord", "f8", (rlen - 1,))])
    elif rtyp == 1940:
        return _abq_dtype(
            ("key", "u4"), ("label", f"S{(rlen - 1) * ftnfil.AWL}")
        )
    else:
        msg = f"Unknown record {rtyp}"
        raise ValueError(msg)


class AbqFil:
    @staticmethod
    def b2str(b):
        return b.decode("ASCII").rstrip()

    def __str__(self):
        return (
            f"{self.path},"
            f" {self.b2str(self.info['date'])}"
            f" {self.b2str(self.info['time'])},"
            f" Abaqus ver. {self.b2str(self.info['ver'])}"
        )

    def __init__(self, path):
        self.path = path
        self.fil = ftnfil.mmfil(path)

        data = self.fil["data"]
        stream = ftnfil.rstream(data)
        pos, rtyp, rlen, rdat = next(stream)

        # 1921: general info
        assert pos == 0 and rtyp == 1921, (pos, rtyp, rlen)
        logger.debug("Collect general info (%.2f)", pos / data.size)
        self.info = rdat.view(ABQR[rtyp])[0]
        pos, rtyp, rlen, rdat = next(stream)

        # 1900, 1990: build element incidences
        assert rtyp == 1900, (pos, rtyp, rlen)
        logger.debug("Collect elm data (%.2f)", pos / data.size)
        self.elm = []
        while rtyp == 1900:
            s_pos, s_rtyp, s_rlen = pos, rtyp, rlen
            pos, rtyp, rlen, rdat = stream.send(())
            mesh = ftnfil.datablock(data, s_pos, pos, s_rlen).view(
                _vlenr(s_rtyp, s_rlen)
            )
            # sometimes abaqus gathers elements of different eltype in
            # the same 1900 record
            for eltyp in np.unique(mesh["eltyp"]):
                mesh_comp = mesh[mesh["eltyp"] == eltyp]
                assert _isunique(mesh_comp["elnum"])
                self.elm.append(mesh_comp)

            ## FIXME: check il 1990 record handling is compatible
            ## with above 1900 multi-element type record
            while rtyp == 1990:  # continuation record
                assert len(self.elm[-1]) == 1
                elnum, eltyp, ninc = self.elm[-1][0]
                ninc = np.append(ninc, rdat.view("i8"))
                self.elm[-1] = np.array(
                    [(elnum, eltyp, ninc)], dtype=_vlenr(1900, len(ninc) + 2)
                )

                pos, rtyp, rlen, rdat = next(stream)

        # 1901: build nodal coordinates
        logger.debug("Collect node data (%.2f)", pos / data.size)
        assert rtyp == 1901, (pos, rtyp, rlen)
        s_pos, s_rtyp, s_rlen = pos, rtyp, rlen
        pos, rtyp, rlen, rdat = stream.send(())
        self.coord = ftnfil.datablock(data, s_pos, pos, s_rlen).view(
            _vlenr(s_rtyp, s_rlen)
        )
        assert _isunique(self.coord["nnum"])

        # 1933, 1934: element sets
        logger.debug("Collect elset data (%.2f)", pos / data.size)
        self.elset = {}
        while rtyp == 1933:
            elset_label = bytes(rdat[0])
            elset_array = array.array("I", rdat[1:].view("=2u4")[..., 0])
            pos, rtyp, rlen, rdat = next(stream)

            while rtyp == 1934:
                elset_array.extend(rdat.view("=2u4")[..., 0])
                pos, rtyp, rlen, rdat = next(stream)

            self.elset[elset_label] = np.array(elset_array)
            assert _isunique(self.elset[elset_label])

        # 1931, 1932: node sets
        logger.debug("Collect nset data (%.2f)", pos / data.size)
        self.nset = {}
        while rtyp == 1931:
            nset_label = bytes(rdat[0])
            nset_array = array.array("I", rdat[1:].view("=2u4")[..., 0])
            pos, rtyp, rlen, rdat = next(stream)

            while rtyp == 1932:
                nset_array.extend(rdat.view("=2u4")[..., 0])
                pos, rtyp, rlen, rdat = next(stream)

            self.nset[nset_label] = np.array(nset_array)

        # 1940: label cross reference
        self.label = {}
        while rtyp == 1940:
            k, v = rdat.view(_vlenr(rtyp, rlen)).item()
            k = f"{k:8d}".encode("ASCII")
            self.label[k] = v
            pos, rtyp, rlen, rdat = next(stream)

        # 1902: active degrees of freedom
        assert rtyp == 1902, (pos, rtyp, rlen)
        self.dof = rdat.view("=2u4")[..., 0]
        pos, rtyp, rlen, rdat = next(stream)

        # 1922: heading
        assert rtyp == 1922, (pos, rtyp, rlen)
        self.heading = bytes(rdat)
        pos, rtyp, rlen, rdat = next(stream)

        # 2001: padding
        if rtyp == 2001:
            pos, rtyp, rlen, rdat = next(stream)
        assert pos % ftnfil.AWR == 0

        # 1501, 1502: surfaces
        logger.debug("Collect surf data (%.2f)", pos / data.size)
        self.rsurf = {}
        self.dsurf = {}
        while rtyp == 1501:
            surf = {}
            name, surf["sdim"], stype, nfacet, nmaster, *masters = rdat.view(
                _vlenr(rtyp, rlen)
            ).item()
            assert 1 <= surf["sdim"] <= 4
            assert 1 <= stype <= 2
            if stype == 1:  # deformable
                self.dsurf[name] = surf
                surf["msurf"] = masters
                assert rlen == 2 + 5 + nmaster
                assert len(surf["msurf"]) == nmaster
            elif stype == 2:  # rigid
                self.rsurf[name] = surf
                # meaning of nmaster not defined in ths case
                # abaqus sets nmasters > 0
                # however *masters should be empty
                assert len(masters) == 0, f"unexpected masters: {masters}"
                assert rlen == 2 + 5
            else:
                assert False, f"unrecognized surface type {stype}"
            pos, rtyp, rlen, rdat = next(stream)

            surf["facet_block"] = []
            assert rtyp == 1502
            while rtyp == 1502:
                s_pos, s_rtyp, s_rlen = pos, rtyp, rlen
                pos, rtyp, rlen, rdat = stream.send(())

                # 1502 format
                # Record key: 1502(S)   Record type: Surface facet
                # Attributes:   1  –  Underlying element number.
                #               2  –  Element face key
                #                     (1–S1, 2–S2, 3–S3, 4–S4, 5–S5, 6–S6,
                #                      7–SPOS, 8–SNEG).
                #               3  –  Number of nodes in facet.
                #               4  –  Node number of the facet's first node.
                #               5  –  Node number of the facet's second node.
                #               6  –  Etc.

                # attribute 3 is redundant and not read, skipped with offset
                assert s_rlen - 3 - 2 > 0
                ## FIXME: Shape-1 fields in dtypes
                dt = np.dtype(
                    {
                        "names": ["elnum", "f_id", "nodes"],
                        "formats": [
                            "i4",
                            "i8",
                            f"({s_rlen-3-2:d},)i8"
                            if (s_rlen - 3 - 2) > 1
                            else "i8",
                        ],
                        "itemsize": 8 * (s_rlen - 2),
                        "offsets": [0, 8, 24],
                    }
                )
                surf["facet_block"].append(
                    ftnfil.datablock(data, s_pos, pos, s_rlen).view(dt)
                )
            if __debug__:
                tfacet = 0
                for f in surf["facet_block"]:
                    assert _issorted(f["elnum"])
                    tfacet += len(f)
            assert tfacet == nfacet, (tfacet, nfacet)

        # 2001: padding
        if rtyp == 2001:
            pos, rtyp, rlen, rdat = next(stream)
        assert pos % ftnfil.AWR == 0

        # hic sunt step data
        logger.debug("Collect step data (%.2f)", pos / data.size)
        assert rtyp == 2000

        step_rec, step_data = ftnfil.incstart(data, pos // ftnfil.AWR)
        self.step = np.frombuffer(step_data, dtype=ABQR[2000])
        self.step_rec = step_rec
        assert len(self.step_rec) == len(self.step) + 1
        logger.debug("Found %d steps", len(self.step))
        for i in range(len(self.step)):
            logger.debug(
                "step data: %d (%#.2f -- %#.2f)",
                i,
                step_rec[i] / len(data),
                step_rec[i + 1] / len(data),
            )

    def get_step(self, istep):
        """get step data"""

        logger.debug("Collect step %d", istep)

        # record keys
        # 2000 - inc start
        # <repeat (0 or more times)>
        #    1911 - element output
        #    <repeat (0 or more times)>
        #        1 - element header
        #        <repeat>
        #            XXX - output records
        #        <end>
        #    <end>
        # <end>
        # 2001 - inc stop

        data = self.fil["data"][
            self.step_rec[istep] : self.step_rec[istep + 1]
        ]
        stream = ftnfil.rstream(data)
        pos, rtyp, rlen, rdat = next(stream)

        # skip first 2000 record
        assert rtyp == 2000, rtyp
        pos, rtyp, rlen, rdat = next(stream)

        # iterate over 1911 records
        while True:
            if rtyp == 2001:
                break
            assert rtyp == 1911, (rtyp, rlen)
            outtyp, outset, outelm = rdat.view(ABQR[1911]).item()

            ## FIXME: implemented only for element output
            if outtyp != 0:
                raise NotImplementedError("only element output is implemented")

            assert outtyp == 0, outtyp  # element output
            logger.debug(
                "data block: elset '%s', eltype '%s'",
                self.b2str(outset),
                self.b2str(outelm),
            )

            pos, rtyp, rlen, rdat = next(stream)
            if rtyp == 1911 or rtyp == 2001:
                logger.debug("data block: empty")
                continue

            assert rtyp == 1, rtyp

            # iterate over "columns" of first "row"
            # meta-data of colums is stored in 'types':
            # types is (rkey, offset, data length)
            # data length is <record length> - <header length>
            # where header is (rlen, rkey) thus of length 2

            types = []
            s_pos = pos
            while True:
                pos, rtyp, rlen, rdat = next(stream)
                if rtyp == 1:
                    break
                types.append((rtyp, pos - s_pos, rlen - 2))
            types.append((-1, pos - s_pos, 0))  # sentinel
            assert types[0][1] == 11  # lenght of rkey 1

            # construct dtype for this output block
            # record key: 1
            dtdict = {
                "names": [
                    "num",
                    "ipnum",
                    "spnum",
                    "loc",
                    "rebarname",
                    "ndi",
                    "nshr",
                    "ndir",
                    "nsfc",
                ],
                "formats": [
                    "i4",
                    "i4",
                    "i4",
                    "i4",
                    "S8",
                    "i4",
                    "i4",
                    "i4",
                    "i4",
                ],
                "itemsize": 8 * types[-1][1],
                "offsets": [16, 24, 32, 40, 48, 56, 64, 72, 80],
            }

            assert dtdict["itemsize"] == 8 * (pos - s_pos)

            for k, o, s in types[:-1]:
                dtdict["names"].append(f"R{k:d}")
                ## FIXME: Shape-1 fields in dtypes
                ## see <https://numpy.org/doc/stable/release/1.17.0-notes.html#shape-1-fields-in-dtypes-won-t-be-collapsed-to-scalars-in-a-future-version>
                ## used to be f"{s:d}f8" but "1f8" was a synonim of "f8" in numpy < 1.17
                ## current implementation explicitly enforces this semantics
                ## in the future it should be f"({s:d},)f8" for all s (i.e. s>=1)
                dtdict["formats"].append(f"({s:d},)f8" if s > 1 else "f8")
                dtdict["offsets"].append(16 + o * 8)

            dt = np.dtype(dtdict)
            logger.debug("data block: %s", dt.names)

            # skip to last data record
            ## FIXME: most of decoding time is spent here!
            logger.debug("data block: iterating to find end record")
            pos, rtyp, rlen, rdat = stream.send((1911, 2001))

            # get data
            logger.debug("data block: getting data")
            r = data.flat[s_pos:pos].view(dt)
            logger.debug("data block loc: %s", np.unique(r["loc"]))

            assert _issorted(r["num"])
            if __debug__:
                for k in ["loc", "ndi", "nshr", "ndir", "nsfc"]:
                    assert (r[k] == r[k][0]).all(), (istep, k)

            logger.debug("data block: done")
            yield StepDataBlock(outtyp, outset, outelm, r)

        return
