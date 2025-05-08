# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

"""
Entry point for 'fil2h5'
"""

__all__ = ["main"]

import json
import sys
from pathlib import Path

import click
import h5py
import numpy as np

from suanpan.abqfil import AbqFil


@click.command()
@click.argument(
    "files", nargs=-1, type=click.Path(exists=True, path_type=Path)
)
def main(files: tuple[Path, ...]) -> None:
    for pth in files:
        try:
            with h5py.File(pth.with_suffix(".h5"), mode="w") as h5:
                abq = AbqFil(path=pth)
                labels = Labels(abq.label)
                _add_meta(h5, abq)
                _add_elements(h5, abq)
                _add_nodes(h5, abq)
                _add_sets(h5, abq, labels)
                _add_surfaces(h5, abq, labels)
                _add_steps(h5, abq)
                click.echo(h5.filename)
        except OSError as exc:
            click.echo(f"{pth}: conversion failed: {exc}", file=sys.stderr)


def _add_meta(h5, abq):
    d = _struct_scalar_to_dict(abq.info)
    h5.attrs["info"] = json.dumps(d)
    h5.attrs["heading"] = abq.heading.decode("ASCII")
    h5.create_dataset(name="dof", data=abq.dof)


def _add_elements(h5, abq):
    elements = h5.create_group(name="elements")
    for arecs in abq.elm:
        element_type = arecs["eltyp"][0]
        assert np.all(element_type == arecs["eltyp"])
        element_type = abq.b2str(element_type)
        assert element_type not in elements, (
            f"Duplicate definition for {element_type}"
        )
        group = elements.create_group(element_type)
        group.create_dataset(name="numbers", data=arecs["elnum"])
        group.create_dataset(name="incidences", data=arecs["ninc"])


def _add_nodes(h5, abq):
    nodes = h5.create_group(name="nodes")
    nodes.create_dataset(name="numbers", data=abq.coord["nnum"])
    nodes.create_dataset(name="coordinates", data=abq.coord["coord"])


def _add_sets(h5, abq, labels):
    for n, d in abq.nset.items():
        h5.create_dataset(name=f"sets/node/{labels(n)}", data=d)
    for n, d in abq.elset.items():
        h5.create_dataset(name=f"sets/element/{labels(n)}", data=d)


def _add_surfaces(h5, abq, labels):
    """save surface definitions (record 1501, 1502)"""

    # FIXME: refactor to avoid repetition
    for n, d in abq.rsurf.items():
        name = labels(n)
        grp = h5.create_group(name=f"surfaces/rigid/{name}")
        grp.attrs["dim"] = d["sdim"]
        for i, fb in enumerate(d["facet_block"]):
            grp.create_dataset(name=f"{i}/elements", data=fb["elnum"])
            grp.create_dataset(name=f"{i}/facets", data=fb["f_id"])
            grp.create_dataset(name=f"{i}/nodes", data=fb["nodes"])
    for n, d in abq.dsurf.items():
        name = labels(n)
        if d["msurf"]:
            grp = h5.create_group(name=f"surfaces/deformable/secondary/{name}")
            grp.attrs["main"] = json.dumps([labels(i) for i in d["msurf"]])
        else:
            grp = h5.create_group(name=f"surfaces/deformable/main/{name}")
        grp.attrs["dim"] = d["sdim"]
        for i, fb in enumerate(d["facet_block"]):
            grp.create_dataset(name=f"{i}/elements", data=fb["elnum"])
            grp.create_dataset(name=f"{i}/facets", data=fb["f_id"])
            grp.create_dataset(name=f"{i}/nodes", data=fb["nodes"])


def _add_steps(h5, abq):
    for i, h in enumerate(abq.step):
        inc = h5.create_group(name=f"results/inc{i:03d}")
        inc.attrs.update(_struct_scalar_to_dict(h))
        for j, k in enumerate(abq.get_step(i)):
            out = inc.create_group(name=f"req{j:03d}")
            out.attrs["type"] = k.flag
            out.attrs["set"] = AbqFil.b2str(k.set)
            out.attrs["eltype"] = AbqFil.b2str(k.eltype)

            if k.flag == 0:
                singleton = ["loc", "rebarname", "ndi", "nshr", "ndir", "nsfc"]
                assert np.all(k.data[singleton] == k.data[singleton][0])
                out.attrs.update(_struct_scalar_to_dict(k.data[singleton][0]))
                shape, dim = _reshape(k.data[["num", "ipnum", "spnum"]])

                dimensions = out.create_group("dimensions", track_order=True)
                dimensions.update(dim)

                data = k.data.reshape(shape, copy=False)
                for n in data.dtype.names:
                    if not n.startswith("R"):
                        continue
                    out.create_dataset(n, data=data[n])
            else:
                msg = f"Data for location = {k.flag} not supported"
                raise NotImplementedError(msg)


def _struct_scalar_to_dict(s):
    d = {}
    for name, (dtype, _) in s.dtype.fields.items():
        d[name] = (
            s[name].item()
            if dtype.char != "S"
            else s[name].item().decode("ASCII")
        )
    return d


class Labels:
    """aux class for translating abaqus label pointers"""

    def __init__(self, labels_map):
        self.labels_map = {k: AbqFil.b2str(v) for k, v in labels_map.items()}

    def __call__(self, key):
        try:
            return self.labels_map[key]
        except KeyError:
            return AbqFil.b2str(key)


def _reshape(v):
    a = v[()]
    names = a.dtype.names
    dim = {}
    lead = ()
    for k in names[:-1]:
        shape = a.shape
        assert len(shape) == len(lead) + 1

        q = a[lead][k]
        assert q.shape == (shape[-1],)
        assert np.ndim(q) == 1
        assert len(q) == shape[-1]

        z = q[0]
        l = 1
        while l < len(q) and q[l] == z:
            l += 1

        if l == 1 and z == 0:
            continue

        shape = shape[:-1] + (-1, l)
        a.shape = shape
        assert np.all(
            a[np.index_exp[0:1] * len(lead) + np.index_exp[:, 0:1]][k] == a[k]
        )
        dim[k] = a[lead + np.index_exp[:, 0]][k]
        lead += (0,)
    # last dimension is special cased
    k = names[-1]
    assert np.all(
        a[np.index_exp[0:1] * len(lead) + np.index_exp[:]][k] == a[k]
    )
    if a.shape[-1] == 1 and a[lead + (0,)][k] == 0:
        # drop last axis
        a = np.squeeze(a, axis=-1)
    else:
        dim[k] = a[(0,) * (np.ndim(a) - 1) + np.index_exp[:]][k]
    assert len(a.shape) == len(dim)
    if __debug__:
        for i, k in enumerate(dim):
            assert a.shape[i] == len(dim[k])
            assert np.all(
                np.reshape(dim[k], (-1,) + (1,) * (np.ndim(a) - 1 - i)) == a[k]
            )
    return a.shape, dim


if __name__ == "__main__":
    main()
