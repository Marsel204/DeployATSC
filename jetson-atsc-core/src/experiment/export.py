from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import threading
import uuid
import zipfile


class ExperimentStore:
    def __init__(self, root, run_id, manifest):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):raise ValueError("Invalid run identity")
        self.directory = Path(root)/run_id
        self.directory.mkdir(parents=True,exist_ok=False)
        self.manifest = dict(manifest, schema_version=1, run_id=run_id,
                             created_utc=datetime.now(timezone.utc).isoformat())
        self.lock = threading.RLock()
        self._write("manifest.json",self.manifest)

    def _write(self, name, data):
        temporary = self.directory/(name+"."+uuid.uuid4().hex+".tmp")
        temporary.write_text(json.dumps(data,indent=2,allow_nan=False))
        temporary.replace(self.directory/name)

    def persist(self, state):
        with self.lock:
            self._write("evidence.json",state)
            for key in ("predictions","phase_events","vehicles","comparisons"):
                lines = "".join(json.dumps(row,allow_nan=False)+"\n" for row in state.get(key,[]))
                temporary=self.directory/(key+"."+uuid.uuid4().hex+".tmp")
                temporary.write_text(lines)
                temporary.replace(self.directory/(key+".jsonl"))

    @staticmethod
    def _csv(rows):
        stream=io.StringIO(newline="")
        fields=sorted({key for row in rows for key in row}) or ["run_id"]
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})
        return stream.getvalue()

    def export(self,state):
        self.persist(state)
        output=io.BytesIO()
        with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("manifest.json",json.dumps(self.manifest,indent=2,allow_nan=False))
            bundle.writestr("evidence.json",json.dumps(state,indent=2,allow_nan=False))
            for key in ("predictions","phase_events","vehicles","comparisons"):
                bundle.writestr(key+".csv",self._csv(state.get(key,[])))
        return output.getvalue()
