# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

"""high level interface"""

import array
import itertools
import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any, NamedTuple

import numpy as np

from . import ftnfil

logging.captureWarnings(True)
logger = logging.getLogger(__name__)


# named tuple for 1911 + datarecords
class StepDataBlock(NamedTuple):
    flag: int
    set: bytes
    eltype: bytes
    data: np.ndarray


def _issorted(v: np.ndarray) -> np.bool:
    """returns true if vector v is sorted"""
    return np.all(v[:-1] <= v[1:])


def _issorted_strict(v: np.ndarray) -> np.bool:
    """returns true if vector v is sorted with no repetitions"""
    return np.all(v[:-1] < v[1:])


def _pad(x: int) -> int:
    """return min multiple of AWL grater than or equal to x"""
    return x + (-x % ftnfil.AWL)


def _abq_dtype(items: Sequence[tuple[str, str | tuple[str, int]]]) -> np.dtype:
    """make abaqus dtype"""

    if not items:
        msg = "The number of items should be grater than zero"
        raise ValueError(msg)

    names, formats = zip(*items, strict=True)
    formats = tuple(map(np.dtype, formats))
    cumsum = tuple(itertools.accumulate(_pad(i.itemsize) for i in formats))
    assert len(names) == len(formats) == len(cumsum)
    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "offsets": (0, *cumsum[:-1]),
            "itemsize": cumsum[-1],
        }
    )


def _record_dtype(rtyp: int, rlen: int) -> np.dtype:
    """return numpy dtype for Abaqus records"""

    items: list[tuple[str, str | tuple[str, int]]]
    match rtyp, rlen - 2:
        case (1501, dlen) if dlen >= 5:
            items = [
                ("name", "S8"),
                ("sdim", "i4"),
                ("stype", "i4"),
                ("nfacet", "i4"),
                ("nmaster", "i4"),
                *((f"msurf{i + 1}", "S8") for i in range(dlen - 5)),
            ]
        case (1900, dlen) if dlen > 2:
            items = [
                ("elnum", "i8"),
                ("eltyp", "S8"),
                ("ninc", ("i8", dlen - 2)),
            ]
        case (1901, dlen) if dlen > 1:
            items = [
                ("nnum", "i8"),
                ("coord", ("f8", dlen - 1)),
            ]
        case (1911, 3):
            items = [
                ("out_type", "i4"),
                ("out_set", "S8"),
                ("out_element", "S8"),
            ]
        case (1921, 7):
            items = [
                ("ver", "S8"),
                ("date", "S16"),
                ("time", "S8"),
                ("nelm", "u4"),
                ("nnod", "u4"),
                ("elsiz", "f8"),
            ]
        case (1940, dlen) if dlen > 1:
            items = [
                ("key", "u4"),
                ("label", f"S{(dlen - 1) * ftnfil.AWL}"),
            ]
        case (2000, 21):
            items = [
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
            ]
        case _:
            msg = f"Unrecognized record: rtype = {rtyp}, rlen = {rlen}"
            raise ValueError(msg)

    dtype = _abq_dtype(items)
    assert dtype.itemsize == (rlen - 2) * ftnfil.AWL, (rtyp, rlen)
    return dtype


class AbqFil:
    @staticmethod
    def b2str(b: bytes) -> str:
        return b.decode("ASCII").rstrip()

    def __str__(self) -> str:
        return (
            f"{self.path},"
            f" {self.b2str(self.info['date'])}"
            f" {self.b2str(self.info['time'])},"
            f" Abaqus ver. {self.b2str(self.info['ver'])}"
        )

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = path
        self.fil = ftnfil.mmfil(path)

        data = self.fil["data"]
        stream = ftnfil.rstream(data)
        pos, rtyp, rlen, rdat = next(stream)

        # 1921: general info
        assert pos == 0 and rtyp == 1921, (pos, rtyp, rlen)
        logger.debug("Collect general info (%.2f)", pos / data.size)
        self.info = rdat.view(_record_dtype(rtyp, rlen))[0]
        pos, rtyp, rlen, rdat = next(stream)

        # 1900, 1990: build element incidences
        assert rtyp == 1900, (pos, rtyp, rlen)
        logger.debug("Collect elm data (%.2f)", pos / data.size)
        elm: dict[bytes, list[np.ndarray]] = {}
        while rtyp == 1900:
            s_pos, s_rtyp, s_rlen = pos, rtyp, rlen
            pos, rtyp, rlen, rdat = stream.send(())
            mesh = ftnfil.datablock(data, s_pos, pos, s_rlen).view(
                _record_dtype(s_rtyp, s_rlen)
            )
            # usually len(np.unique(mesh["eltyp"]) == 1, but sometimes
            # we have consecutive element blocks with the same number
            # of nodes, i.e. same rlen.
            for eltyp in np.unique(mesh["eltyp"]):
                mesh_comp = mesh[mesh["eltyp"] == eltyp]
                assert _issorted_strict(mesh_comp["elnum"])
                elm.setdefault(eltyp, []).append(mesh_comp)

            ## FIXME: 1990 record handling not tested!
            while rtyp == 1990:  # continuation record
                assert len(elm[eltyp][-1]) == 1
                elnum, _, ninc = elm[eltyp][-1][0]
                assert _ == eltyp
                ninc = np.append(ninc, rdat.view("i8"))
                elm[eltyp][-1] = np.array(
                    [(elnum, eltyp, ninc)],
                    dtype=_record_dtype(1900, len(ninc) + 2),
                )
                pos, rtyp, rlen, rdat = next(stream)

        # fuse all homogeneous mesh components
        self.elm = [np.concat(elm.pop(eltyp)) for eltyp in list(elm)]
        assert elm == {}
        for v in self.elm:
            assert np.all(v["eltyp"] == v["eltyp"][0])
            assert _issorted_strict(v["elnum"])
            if np.any(v["elnum"] - v["elnum"][0] != np.arange(len(v))):
                logger.warning(
                    "Element numbers are not consecutive: %s",
                    self.b2str(v["eltyp"][0]),
                )

        # 1901: build nodal coordinates
        logger.debug("Collect node data (%.2f)", pos / data.size)
        assert rtyp == 1901, (pos, rtyp, rlen)
        s_pos, s_rtyp, s_rlen = pos, rtyp, rlen
        pos, rtyp, rlen, rdat = stream.send(())
        self.coord = ftnfil.datablock(data, s_pos, pos, s_rlen).view(
            _record_dtype(s_rtyp, s_rlen)
        )
        assert _issorted_strict(self.coord["nnum"])

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
            assert _issorted_strict(self.elset[elset_label])

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
            k, v = rdat.view(_record_dtype(rtyp, rlen)).item()
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
                _record_dtype(rtyp, rlen)
            ).item()
            assert 1 <= surf["sdim"] <= 4
            assert 1 <= stype <= 2
            if stype == 1:  # deformable
                self.dsurf[name] = surf
                surf["msurf"] = masters
                # for deformable surfaces 'nmaster' is the number of associated
                # master surfaces
                assert rlen == 2 + 5 + nmaster
                assert len(surf["msurf"]) == nmaster
            elif stype == 2:  # rigid
                self.rsurf[name] = surf
                # for rigid surfaces 'nmaster' is the reference node label
                surf["reference_node"] = nmaster
                assert rlen == 2 + 5
                assert len(masters) == 0, f"unexpected masters: {masters}"
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
                # Attributes:   1  -  Underlying element number.
                #               2  -  Element face key
                #                     (1-S1, 2-S2, 3-S3, 4-S4, 5-S5, 6-S6,
                #                      7-SPOS, 8-SNEG).
                #               3  -  Number of nodes in facet.
                #               4  -  Node number of the facet's first node.
                #               5  -  Node number of the facet's second node.
                #               6  -  Etc.

                # attribute 3 is redundant and not read, skipped with offset
                assert s_rlen - 3 - 2 > 0
                dt = np.dtype(
                    {
                        "names": ["elnum", "f_id", "nodes"],
                        "formats": ["i4", "i8", f"({s_rlen - 3 - 2:d},)i8"],
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
        self.step = np.frombuffer(step_data, dtype=_record_dtype(rtyp, rlen))
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

    def get_step(self, istep: int) -> Iterator[StepDataBlock]:
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

        data = self.fil["data"][self.step_rec[istep] : self.step_rec[istep + 1]]
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
            outtyp, outset, outelm = rdat.view(_record_dtype(rtyp, rlen)).item()

            ## FIXME: implemented only for element output
            if outtyp != 0:
                msg = "only element output is implemented"
                raise NotImplementedError(msg)

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
            dtdict: dict[str, Any] = {
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
                dtdict["formats"].append(f"({s:d},)f8")
                dtdict["offsets"].append(16 + o * 8)

            dt = np.dtype(dtdict)  # type: ignore[call-overload]
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
                    assert np.all(r[k] == r[k][0]), (istep, k)

            logger.debug("data block: done")
            yield StepDataBlock(outtyp, outset, outelm, r)

        return
