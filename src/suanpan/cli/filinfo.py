# SPDX-FileCopyrightText: 2025 Stefano Miccoli <stefano.miccoli@polimi.it>
#
# SPDX-License-Identifier: MIT

import argparse
import sys
from typing import Any

try:
    import yaml
except ImportError:
    print(
        "'filinfo' command requires the optional dependency 'filinfo'",
        file=sys.stderr,
    )
    sys.exit(2)

from suanpan.abqfil import AbqFil, StepDataBlock, StepDataBlockElement

b2str = AbqFil.b2str


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="filinfo",
        description="Outputs a YAML summary of the .fil files",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("filename", nargs="*")
    args = parser.parse_args()

    for pth in args.filename:
        try:
            abq = AbqFil(pth)
        except OSError as exc:
            print(f"{exc}", file=sys.stderr)
            continue
        except ValueError as exc:
            print(f"Invalid .fil file: {exc}", file=sys.stderr)
            continue

        info = {
            "path": abq.path,
            "version": b2str(abq.info["ver"]),
            "date": f"{b2str(abq.info['date'])} {b2str(abq.info['time'])}",
            "heading": b2str(abq.heading) or None,
            "nodes": abq.info["nnod"].item(),
            "elements": {b2str(v["eltyp"][0]): len(v) for v in abq.elm}
            | {"total": abq.info["nelm"].item()},
            "frames": [
                {
                    "step": s["step"].item(),
                    "increment": s["incr"].item(),
                    "time": s["ttime"].item(),
                    "subheading": b2str(s["subheading"]) or None,
                }
                for s in abq.step
            ],
        }

        if args.verbose:
            for i, frame_info in enumerate(info["frames"]):
                data = frame_info["data"] = []
                for db in abq.get_step(i):
                    db_info: dict[str, Any] = {
                        "flag": db.flag,
                        "set": b2str(abq.label.get(db.set, db.set)) or None,
                    }
                    match db:
                        case StepDataBlock(flag=0):
                            assert isinstance(db, StepDataBlockElement)
                            db_info |= {
                                "eltype": b2str(db.eltype),
                                "location": db.data["loc"][0].item(),
                                "records": [
                                    r
                                    for r in db.data.dtype.names
                                    if r.startswith("R")
                                ],
                            }
                        case _:
                            pass
                    data.append(db_info)

        print(
            yaml.safe_dump(
                info,
                default_flow_style=False,
                explicit_start=True,
                explicit_end=False,
                sort_keys=False,
                allow_unicode=False,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
